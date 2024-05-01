use derive_builder::Builder;
use macro_rules_attribute::derive;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    mem,
    slice::Windows,
};
use tokenizers::{models::wordpiece::WordPiece, models::bpe::BPE};
use tokenizers::parallelism::*;
use tokenizers::{AddedToken, ModelWrapper, Result, Trainer};

use crate::pre_tokenizers::SmirkPreTokenizer;

type Pair = (u32, u32);

#[derive(PartialEq, Debug)]
struct Word {
    glyphs: Vec<u32>,
}

impl Word {
    fn windows(&self, size: usize) -> Windows<'_, u32> {
        self.glyphs.windows(size)
    }

    // Replace a Pair of tokens with a new token (id)
    // Return a HashMap of pair => ±n for the impact of this merge on pair_counts
    fn merge(&mut self, pair: Pair, id: u32) -> HashMap<Pair, i64> {
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
#[derive(Builder)]
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
    // How to split words into glyphs
    pub pattern: SmirkPreTokenizer,
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
            pattern: SmirkPreTokenizer::new(true),
            special_tokens: Vec::new(),
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
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        let mut alphabet: HashMap<String, usize> = HashMap::new();
        let pattern = &self.pattern;
        for (word, count) in wc {
            for token in pattern.split(word) {
                alphabet
                    .entry(token)
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
    fn add_special_tokens(&self, w2id: &mut HashMap<String, u32>, id2w: &mut Vec<String>) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    fn tokenize_words(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) -> (Vec<Word>, Vec<i64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<i64> = Vec::with_capacity(wc.len());
        for (word, count) in wc {
            counts.push(*count as i64);
            let symbol_ids = self.pattern.split(word).into_iter().map(|symbol| {
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
        model: &mut ModelWrapper,
    ) -> Result<Vec<AddedToken>> {
        // Setup initial alphabet
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        self.compute_alphabet(&word_counts, &mut word_to_id, &mut id_to_word);
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        // Tokenize words, returning word_counts => (Vec, Vec)
        let (words, counts) = self.tokenize_words(&word_counts, &mut word_to_id, &mut id_to_word);
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

        // Tabulate merges
        let merges: Vec<(String, String)> = merges
            .into_iter()
            .map(|pair| {
                let left_token = &id_to_word[pair.0 as usize];
                let right_token = &id_to_word[pair.1 as usize];
                (left_token.to_owned(), right_token.to_owned())
            })
            .collect();

        // Construct New Model
        let new_model = match model {
            ModelWrapper::BPE(bpe) => {
                let mut builder = BPE::builder();
                if let Some(suffix) = bpe.end_of_word_suffix.as_ref() {
                    builder = builder.end_of_word_suffix(suffix.to_owned());
                }
                if let Some(prefix) = bpe.continuing_subword_prefix.as_ref() {
                    builder = builder.continuing_subword_prefix(prefix.to_owned());
                }
                if let Some(unk_token) = bpe.unk_token.as_ref() {
                    builder = builder.unk_token(unk_token.to_owned());
                }
                if let Some(dropout) = bpe.dropout {
                    builder = builder.dropout(dropout);
                }
                let new_model = builder
                    .fuse_unk(bpe.fuse_unk)
                    .byte_fallback(bpe.byte_fallback)
                    .vocab_and_merges(word_to_id, merges)
                    .build()
                    .unwrap();
                Ok(ModelWrapper::BPE(new_model))
            }
            ModelWrapper::WordPiece(wp) => {
                let new_model = WordPiece::builder()
                    .unk_token(wp.unk_token.to_owned())
                    .continuing_subword_prefix(wp.continuing_subword_prefix.to_owned())
                    .max_input_chars_per_word(wp.max_input_chars_per_word)
                    .vocab(word_to_id)
                    .build()
                    .unwrap();
                Ok(ModelWrapper::from(new_model))
            },
            ModelWrapper::WordLevel(_) => {
                let new_model = BPE::builder()
                    .vocab_and_merges(word_to_id, merges)
                    .build()
                    .unwrap();
                Ok(ModelWrapper::BPE(new_model))
            }
            _ => Err(()),
        }
        .unwrap();
        let _ = mem::replace(model, new_model);

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for GpeTrainer {
    type Model = ModelWrapper;

    // Don't use this, use `GPETrainer.train_from_files` directly
    fn train(&self, model: &mut ModelWrapper) -> Result<Vec<AddedToken>> {
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
    use crate::pre_tokenizers::{PreTokenizerWrapper, SplitStructure};
    use std::path::PathBuf;
    use tokenizers::{
        normalizers::Strip, DecoderWrapper, Model, PostProcessorWrapper, TokenizerBuilder,
        TokenizerImpl,
    };

    #[test]
    fn test_init() {
        let trainer = GpeTrainer::builder()
            .alphabet(["C".to_string(), "Cl".to_string(), "B".to_string()].into())
            .build()
            .unwrap();
        assert!(trainer.alphabet.contains("C"));
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
    fn test_train() {
        let word_counts: HashMap<String, u64> = [
            ("CSCCSCCS".into(), 1),
            ("CCCCC".into(), 4),
            ("CCSC".into(), 1),
            ("CC".into(), 1),
            ("CS".into(), 3),
        ]
        .into();
        let trainer = GpeTrainer::builder().vocab_size(5).build().unwrap();
        let mut model = ModelWrapper::BPE(BPE::default());
        let _ = trainer.do_train(&word_counts, &mut model);

        let expected_vocab: HashMap<String, u32> = [
            ("C".into(), 0),
            ("S".into(), 1),
            ("CC".into(), 2),
            ("CS".into(), 3),
            ("CCC".into(), 4),
        ]
        .into();
        assert_eq!(model.get_vocab(), expected_vocab);
    }

    #[test]
    fn test_tokenizer() {
        let pt = PreTokenizerWrapper::SplitStructure(SplitStructure::default());
        let mut tokenizer: TokenizerImpl<
            ModelWrapper,
            Strip,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = TokenizerBuilder::default()
            .with_model(BPE::default().into())
            .with_pre_tokenizer(Some(pt))
            .build()
            .unwrap();
        let mut trainer = GpeTrainer::default();
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/smiles.txt");
        let files: Vec<String> = vec![test_file.to_string_lossy().into()];
        let _ = tokenizer.train_from_files(&mut trainer, files);
    }
}
