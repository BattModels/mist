use derive_builder::Builder;
use either::Either;
use macro_rules_attribute::derive;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    slice::Windows,
};
use tokenizers::parallelism::{MaybeParallelBridge, MaybeParallelRefIterator};
use tokenizers::{AddedToken, Result, Trainer};

use super::model::GPE;

type Pair = (u32, u32);

#[derive(PartialEq, Debug)]
pub struct Word {
    glyphs: Vec<u32>,
}

impl Word {
    pub fn windows(&self, size: usize) -> Windows<'_, u32> {
        self.glyphs.windows(size)
    }

    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    // Replace a Pair of tokens with a new token (id)
    // Return a HashMap of pair => ±n for the impact of this merge on pair_counts
    pub fn merge(&mut self, pair: Pair, id: u32) -> HashMap<Pair, i64> {
        let mut changes: HashMap<Pair, i64> = HashMap::new();
        let mut ldx = 0;
        let word = &mut self.glyphs;
        for rdx in 1..word.len() {
            let cur_pair = (word[ldx], word[rdx]);
            if cur_pair == pair {
                *changes.entry(cur_pair).or_insert(0) -= 1;
                if ldx != 0 {
                    // Update Left-side Pair Count
                    *changes.entry((word[ldx - 1], word[ldx])).or_insert(0) -= 1;
                    *changes.entry((word[ldx - 1], id)).or_insert(0) += 1;
                }
                if rdx + 1 < word.len() {
                    // Update Right-side Pair Count
                    *changes.entry((word[rdx], word[rdx + 1])).or_insert(0) -= 1;
                    *changes.entry((id, word[rdx + 1])).or_insert(0) += 1;

                    // Move over a glyph across the ldx - rdx gap
                    word[ldx + 1] = word[rdx + 1];
                }
                word[ldx] = id;
            } else {
                ldx += 1;
                word[ldx] = word[rdx];
            }
        }
        word.drain(ldx + 1..word.len());
        changes
    }
}

impl FromIterator<u32> for Word {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let glyphs: Vec<u32> = iter.into_iter().collect();
        Word { glyphs }
    }
}

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.count != other.count {
            return self.count.cmp(&other.count);
        }

        // Resolve ties in favor of smaller pairs
        other.pair.cmp(&self.pair)
    }
}

// Glyph Pair Encoding - BPE but supports multi-character "glyphs"
#[derive(Builder, Debug, Deserialize, Serialize, Clone)]
#[builder(default)]
pub struct GpeTrainer {
    // the min frequency of a pair to produce a merge operation
    pub min_frequency: u64,
    // the target vocabulary size
    pub vocab_size: usize,
    // the initial alphabet
    pub alphabet: HashSet<String>,
    // limit the size of the initial alphabet
    pub limit_alphabet: Option<usize>,
    // Special tokens to include in the vocab
    pub special_tokens: Vec<AddedToken>,
    // Should bracket ([, ]) be candidates for merges
    pub merge_brackets: bool,
    // Internal Map for tracking word counts
    word_counts: HashMap<String, u64>,
}

impl Default for GpeTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            vocab_size: 1024,
            alphabet: HashSet::new(),
            limit_alphabet: None,
            special_tokens: Vec::new(),
            merge_brackets: false,
            word_counts: HashMap::new(),
        }
    }
}

impl GpeTrainer {
    pub fn builder() -> GpeTrainerBuilder {
        GpeTrainerBuilder::default()
    }

    #[allow(dead_code)]
    fn new(min_frequency: u64, vocab_size: usize, alphabet: HashSet<String>) -> Self {
        Self {
            min_frequency,
            vocab_size,
            alphabet,
            ..Default::default()
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        model: &GPE,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        let mut alphabet: HashMap<String, usize> = HashMap::new();
        for (word, count) in wc {
            for glyph in model.tokenize.split(word) {
                alphabet
                    .entry(glyph)
                    .and_modify(|c| *c += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Add the initial alphabet
        self.alphabet.iter().for_each(|glyph| {
            alphabet
                .entry(glyph.to_owned())
                .and_modify(|c| *c = std::usize::MAX)
                .or_insert(std::usize::MAX);
        });

        // Sort the alphabet and populate w2id and id2w
        let mut alphabet = alphabet.into_iter().collect::<Vec<_>>();

        // Truncate alphabet, if required, by removing the most uncommon glyphs
        if let Some(limit) = self.limit_alphabet {
            if alphabet.len() > limit {
                let n_remove = alphabet.len() - limit;
                alphabet.sort_unstable_by_key(|k| k.1);
                alphabet.drain(..n_remove);
            }
        }

        // Sort for determinism
        alphabet.sort_unstable_by_key(|k| k.0.to_owned());
        for (glyph, _) in alphabet {
            if !w2id.contains_key(&glyph) {
                id2w.push(glyph.clone());
                w2id.insert(glyph, (id2w.len() - 1) as u32);
            }
        }
    }
    ///
    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(
        &self,
        model: &GPE,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        if !w2id.contains_key(&model.unk_token) {
            id2w.push(model.unk_token.to_owned());
            w2id.insert(model.unk_token.to_owned(), (id2w.len() - 1) as u32);
        }
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    fn tokenize_words(
        &self,
        model: &GPE,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) -> (Vec<Word>, Vec<i64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<i64> = Vec::with_capacity(wc.len());
        for (word, count) in wc {
            counts.push(*count as i64);
            let token_iter = model.tokenize.split(word).into_iter();

            let iter = if !self.merge_brackets {
                Either::Left(token_iter.filter(|s| !(s == "[" || s == "]")))
            } else {
                Either::Right(token_iter)
            };

            let symbol_ids = iter.map(|symbol| {
                w2id.get(&symbol)
                    .map(|v| v.to_owned())
                    .or_else(|| {
                        let id = id2w.len() as u32;
                        id2w.push(symbol.to_string());
                        w2id.insert(symbol, id);
                        Some(id)
                    })
                    .unwrap()
            });
            words.push(symbol_ids.collect());
        }
        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[i64],
    ) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .filter(|(_, word)| word.len() >= 2) // Words shorter than 2 have no pair
            .map(|(i, word)| {
                let mut pair_count = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for token_pair in word.windows(2) {
                    let pair: Pair = (*token_pair.get(0).unwrap(), *token_pair.get(1).unwrap());
                    let count = counts[i];
                    pair_count
                        .entry(pair)
                        .and_modify(|c| *c += count)
                        .or_insert(count);
                    where_to_update
                        .entry(pair)
                        .and_modify(|s| {
                            s.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut s = HashSet::new();
                            s.insert(i);
                            s
                        });
                }
                (pair_count, where_to_update)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_count, mut where_to_update), (pc, p2w)| {
                    for (pair, count) in pc {
                        pair_count
                            .entry(pair)
                            .and_modify(|c| *c += count)
                            .or_insert(count);
                        let words = p2w.get(&pair).unwrap();
                        where_to_update
                            .entry(pair)
                            .and_modify(|s| {
                                words.iter().for_each(|w| {
                                    s.insert(*w);
                                });
                            })
                            .or_insert(words.clone());
                    }
                    (pair_count, where_to_update)
                },
            )
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut GPE,
    ) -> Result<Vec<AddedToken>> {
        // Setup initial alphabet
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        self.add_special_tokens(&model, &mut word_to_id, &mut id_to_word);
        self.compute_alphabet(&model, &word_counts, &mut word_to_id, &mut id_to_word);

        // Save vocab without merges
        let vocab = word_to_id.to_owned();

        // Tokenize words, returning word_counts => (Vec, Vec)
        let (words, counts) =
            self.tokenize_words(&model, &word_counts, &mut word_to_id, &mut id_to_word);
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts);

        // Build a priority queue of merges
        let mut queue = BinaryHeap::new();
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            queue.push(Merge {
                pair,
                count: count.try_into().unwrap(),
                pos,
            });
        });

        let mut merges: Vec<Pair> = Vec::new();
        loop {
            // Stop if the vocab is large enough, or we have no merges left
            if (word_to_id.len() >= self.vocab_size) || queue.is_empty() {
                break;
            }

            // Pop a pair from the queue
            let mut top = queue.pop().unwrap();

            if top.count != pair_counts[&top.pair] as u64 {
                // Previous merge reduced the count, update
                top.count = pair_counts[&top.pair] as u64;
                if top.count != 0 {
                    queue.push(top);
                }
                continue;
            }

            // Check if pair meets threshold for merge
            if top.count < 1 || top.count < self.min_frequency {
                continue;
            }

            // Create a new token for the most frequently occurring pair
            let left_token = &id_to_word[top.pair.0 as usize];
            let right_token = &id_to_word[top.pair.1 as usize];
            let new_token = format!("{}{}", left_token, right_token);
            id_to_word.push(new_token.clone());
            let new_token_id = (id_to_word.len() - 1) as u32;
            word_to_id.insert(new_token, new_token_id);
            merges.push(top.pair);

            // Update words with new token, recording which pairs to update
            let changes = top
                .pos
                .maybe_par_iter()
                .map(|&i| {
                    let word = &words[i] as *const _ as *mut Word;
                    unsafe { ((*word).merge(top.pair, new_token_id), i) }
                })
                .collect::<Vec<_>>();

            // Update pair_counts with changes
            for (change, iw) in changes {
                // Update pair_count
                let word_count = counts[iw];
                change.iter().for_each(|(pair, delta)| {
                    let count = *delta * (word_count as i64);
                    let _ = *pair_counts
                        .entry(*pair)
                        .and_modify(|c| *c += count)
                        .or_insert(count);

                    // If count is positive, may have a new word to update
                    if count > 0 {
                        where_to_update
                            .entry(*pair)
                            .and_modify(|s| {
                                s.insert(iw);
                            })
                            .or_insert({
                                let mut s = HashSet::new();
                                s.insert(iw);
                                s
                            });
                    }
                });
            }
            // Update the queue with new pairs
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair] as u64;
                queue.push(Merge { pair, count, pos });
            });
        }

        // Update Model
        model.with_vocab_and_merges(vocab, merges);
        Ok(self.special_tokens.clone())
    }
}

impl Trainer for GpeTrainer {
    type Model = GPE;

    // Don't use this, use `GPETrainer.train_from_files` directly
    fn train(&self, model: &mut GPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.word_counts, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        false
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.word_counts = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrapper::PreTokenizerWrapper;
    use crate::{gpe::GPE, pre_tokenizers::split_structure};
    use std::path::PathBuf;
    use tokenizers::Model;
    use tokenizers::{
        normalizers::Strip, DecoderWrapper, PostProcessorWrapper, TokenizerBuilder, TokenizerImpl,
    };

    #[test]
    fn test_trainer() {
        let word_counts: HashMap<String, u64> = [
            ("C", 4),
            ("CSCCSCCS", 2),
            ("CCSC", 1),
            ("[C@H]", 2),
            ("(", 3),
            (")", 4),
            ("CS", 3),
        ]
        .into_iter()
        .map(|(s, c)| (s.into(), c))
        .collect();
        let trainer = GpeTrainer::default();
        let mut model = GPE::default();
        assert_eq!(model.unk_token, "[UNK]");
        let _ = trainer.do_train(&word_counts, &mut model);

        let expected_vocab: HashMap<String, u32> = [
            ("[UNK]", 0),
            ("(", 1),
            (")", 2),
            ("@", 3),
            ("C", 4),
            ("H", 5),
            ("S", 6),
            ("[", 7),
            ("]", 8),
        ]
        .into_iter()
        .map(|(s, c)| (s.into(), c))
        .collect();
        let expected_merges: Vec<Pair> =
            [(4, 6), (4, 9), (3, 5), (4, 11), (9, 10), (13, 10), (10, 4)].into();
        assert_eq!(model.vocab, expected_vocab);
        assert_eq!(model.merges, expected_merges);
        assert_eq!(model.get_vocab_size(), 16);
        assert_eq!(model.id_to_token(9).unwrap(), "CS");
        assert_eq!(model.id_to_token(10).unwrap(), "CCS");
        assert_eq!(model.id_to_token(11).unwrap(), "@H");
        assert_eq!(model.id_to_token(12).unwrap(), "C@H");
        assert_eq!(model.id_to_token(13).unwrap(), "CSCCS");
        assert_eq!(model.id_to_token(14).unwrap(), "CSCCSCCS");
        assert_eq!(model.id_to_token(15).unwrap(), "CCSC");
    }

    #[test]
    fn test_merge_change() {
        let mut word = Word {
            glyphs: [0, 1, 3, 4, 5].to_vec(),
        };
        let changes = word.merge((1, 3), 6);
        assert_eq!(word.glyphs, [0, 6, 4, 5]);
        let expect: HashMap<Pair, i64> = HashMap::from([
            ((0, 1), -1),
            ((1, 3), -1),
            ((0, 6), 1),
            ((3, 4), -1),
            ((6, 4), 1),
        ]);
        assert_eq!(changes, expect);
    }

    #[test]
    fn test_double_merge() {
        let mut word = Word {
            glyphs: [0, 1, 3, 1, 3].to_vec(),
        };
        let changes = word.merge((1, 3), 6);
        assert_eq!(word.glyphs, [0, 6, 6]);
        let expect: HashMap<Pair, i64> = HashMap::from([
            ((0, 1), -1),
            ((1, 3), -2),
            ((3, 1), -1),
            ((0, 6), 1),
            ((6, 6), 1),
            ((6, 1), 0),
        ]);
        assert_eq!(changes, expect);
    }

    #[test]
    fn test_merge_nochange() {
        let mut word = Word {
            glyphs: [0, 1, 3, 4, 1, 3, 5].to_vec(),
        };
        let changes = &word.merge((1, 7), 6);
        assert_eq!(word.glyphs, [0, 1, 3, 4, 1, 3, 5]);
        assert!(changes.len() == 0);
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer: TokenizerImpl<
            GPE,
            Strip,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = TokenizerBuilder::default()
            .with_model(GPE::default())
            .with_pre_tokenizer(Some(split_structure().into()))
            .build()
            .unwrap();
        let mut trainer = GpeTrainer::default();
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/smiles.txt");
        let files: Vec<String> = vec![test_file.to_string_lossy().into()];
        let _ = tokenizer.train_from_files(&mut trainer, files).unwrap();
        assert!(tokenizer
            .get_vocab(true)
            .contains_key(&tokenizer.get_model().unk_token))
    }
}
