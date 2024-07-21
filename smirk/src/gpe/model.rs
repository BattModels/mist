use crate::pre_tokenizers::SmirkPreTokenizer;

use super::GpeTrainer;
use derive_builder::Builder;
use serde::{
    de::{MapAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::{Model, PreTokenizedString, PreTokenizer, Result, Token};
use tokenizers::{OffsetReferential, OffsetType};

type Pair = (u32, u32);

// Glyph Pair Encoding - BPE but support multi-character "glyphs"
#[derive(Debug, Builder, Clone, PartialEq, Serialize)]
#[builder(build_fn(skip))]
pub struct GPE {
    // Token used to check represent unknown characters
    pub unk_token: String,
    // Pattern used to split words into glyphs
    pub tokenize: SmirkPreTokenizer,
    // Map of Glyphs to ids, does not include merged glyphs
    #[serde(serialize_with = "ordered_map")]
    pub vocab: HashMap<String, u32>,
    // Ordered list of Glyph ids pairs to merge
    pub merges: Vec<Pair>,
    // Index of the first merged glyph
    #[builder(setter(skip))]
    #[serde(skip_serializing)]
    merge_offset: u32,
    // Inverse mapping from ids to glyphs, includes merged glyphs
    #[builder(setter(skip))]
    #[serde(skip_serializing)]
    vocab_r: HashMap<u32, String>,
}

// Serialize Vocab in id order
fn ordered_map<S, K: Ord + Serialize, V: Serialize>(
    value: &HashMap<K, V>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let ordered: BTreeMap<_, _> = value.iter().collect();
    ordered.serialize(serializer)
}

impl Default for GPE {
    fn default() -> Self {
        let unk_token = "[UNK]".to_string();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert(unk_token.to_owned(), 0);
        GPEBuilder::default()
            .unk_token(unk_token)
            .vocab(vocab)
            .build()
            .unwrap()
    }
}

impl GPEBuilder {
    fn build(&self) -> Result<GPE> {
        let vocab = self.vocab.to_owned().unwrap_or_default();
        let merges = self.merges.to_owned().unwrap_or_default();
        let merge_offset = 1 + *vocab.values().max().unwrap_or(&0);
        let vocab_r = GPE::build_vocab_r(&vocab, &merges, merge_offset)?;
        let tokenize = match &self.tokenize {
            Some(x) => x.to_owned(),
            None => SmirkPreTokenizer::default(),
        };

        Ok(GPE {
            unk_token: self.unk_token.to_owned().expect("Missing unk_token"),
            tokenize,
            vocab,
            vocab_r,
            merges,
            merge_offset,
        })
    }
}

impl From<WordLevel> for GPE {
    fn from(value: WordLevel) -> Self {
        let mut obj = Self {
            unk_token: value.unk_token.to_owned(),
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            merges: Vec::new(),
            merge_offset: 0,
            tokenize: SmirkPreTokenizer::default(),
        };
        let mut vocab = value.get_vocab().to_owned();
        let new_id = vocab.len() as u32;
        vocab.entry(value.unk_token).or_insert(new_id);
        obj.with_vocab_and_merges(vocab, Vec::new());
        obj
    }
}

impl GPE {
    fn build_vocab_r(
        vocab: &HashMap<String, u32>,
        merges: &Vec<Pair>,
        merge_offset: u32,
    ) -> Result<HashMap<u32, String>> {
        let mut vocab_r: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.to_owned())).collect();
        for (pair, id) in merges.into_iter().zip(merge_offset..) {
            let left = vocab_r
                .get(&pair.0)
                .ok_or_else(|| format!("unknown token {}", pair.0))?;
            let right = vocab_r
                .get(&pair.1)
                .ok_or_else(|| format!("unknown token {}", pair.1))?;
            vocab_r.insert(id, format!("{}{}", left, right));
        }
        Ok(vocab_r)
    }

    pub fn with_vocab_and_merges(&mut self, vocab: HashMap<String, u32>, merges: Vec<(u32, u32)>) {
        self.vocab = vocab;
        self.merges = merges;
        self.merge_offset = 1 + *self.vocab.values().max().unwrap_or(&0);
        self.vocab_r = GPE::build_vocab_r(&self.vocab, &self.merges, self.merge_offset).unwrap();
    }

    fn tokenize_glyphs(&self, sequence: &str) -> Result<Vec<Token>> {
        let vocab = &self.vocab;
        let mut splits = PreTokenizedString::from(sequence);
        let _ = self.tokenize.pre_tokenize(&mut splits)?;

        let unk_token_id = *self
            .vocab
            .get(&self.unk_token)
            .expect("Unknown Token not in vocab");

        let tokens: Vec<Token> = splits
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .iter()
            .map(|(s, o, _)| {
                let id = *vocab.get(*s).unwrap_or(&unk_token_id);
                Token {
                    id,
                    value: self.vocab_r[&id].to_owned(),
                    offsets: *o,
                }
            })
            .collect();
        Ok(tokens)
    }

    fn apply_merge(&self, tokens: &mut Vec<Token>, pair: (u32, u32), id: u32) -> Result<()> {
        let mut ldx = 0;
        for rdx in 1..tokens.len() {
            let cur_pair = (tokens[ldx].id, tokens[rdx].id);
            if cur_pair == pair {
                tokens[ldx] = Token {
                    id,
                    value: self.vocab_r.get(&id).unwrap().to_string(),
                    offsets: (tokens[ldx].offsets.0, tokens[rdx].offsets.1),
                };
            } else {
                ldx += 1;
                if ldx != rdx {
                    tokens[ldx] = tokens[rdx].clone();
                }
            }
        }
        tokens.drain(ldx + 1..tokens.len());
        Ok(())
    }

    fn merge_tokens(&self, tokens: &mut Vec<Token>) -> Result<()> {
        for (pair, id) in self.merges.iter().zip(self.merge_offset..) {
            self.apply_merge(tokens, *pair, id)?;
        }
        Ok(())
    }
}

impl Model for GPE {
    type Trainer = GpeTrainer;

    fn save(
        &self,
        folder: &std::path::Path,
        prefix: Option<&str>,
    ) -> tokenizers::Result<Vec<std::path::PathBuf>> {
        let name = match prefix {
            Some(prefix) => format!("{}-gpe.json", prefix),
            None => "gpe.json".to_string(),
        };
        let model_file = folder.join(name);
        let fid = File::open(&model_file).unwrap();
        let _ = serde_json::to_writer(fid, self);
        Ok(vec![model_file])
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }

        // split into glyphs
        let mut word: Vec<Token> = self.tokenize_glyphs(&sequence)?;

        // Merge glyphs, skipping if there's only one
        if word.len() > 1 {
            self.merge_tokens(&mut word)?;
        }
        Ok(word)
    }

    fn get_trainer(&self) -> <Self as Model>::Trainer {
        GpeTrainer::default()
    }

    // Accessors
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.to_owned()
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab.len() + self.merges.len()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }
}

impl<'de> Deserialize<'de> for GPE {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "GPE",
            &["unk_token", "tokenize", "vocab", "merges"],
            GPEVisitor,
        )
    }
}

struct GPEVisitor;
impl<'de> Visitor<'de> for GPEVisitor {
    type Value = GPE;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct GPE")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = GPEBuilder::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token" => {
                    if let Some(x) = map.next_value()? {
                        builder.unk_token(x);
                    }
                }
                "tokenize" => {
                    if let Some(x) = map.next_value()? {
                        builder.tokenize(x);
                    }
                }
                "vocab" => {
                    if let Some(x) = map.next_value()? {
                        builder.vocab(x);
                    }
                }
                "merges" => {
                    if let Some(x) = map.next_value()? {
                        builder.merges(x);
                    }
                }
                "type" => match map.next_value()? {
                    "GPE" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"GPE",
                        ))
                    }
                },
                _ => {}
            };
        }
        Ok(builder.build().unwrap())
    }
}

#[cfg(test)]
mod test {
    use tokenizers::{Model, Token};

    use super::GPE;
    use std::collections::HashMap;

    #[test]
    fn test_default() {
        let model = GPE::default();
        assert_eq!(model.unk_token, "[UNK]");
        assert!(model.vocab.contains_key(&model.unk_token));
    }

    fn check_tokenize<'a>(model: &GPE, smile: &str, expect: Vec<&'a str>, ids: Vec<u32>) {
        let code = model.tokenize(smile).unwrap();
        let tokens: Vec<String> = code.iter().map(|t| t.value.to_owned()).collect();
        let token_ids: Vec<u32> = code.iter().map(|t| t.id).collect();
        assert_eq!(
            tokens, expect,
            "{} -> {:?}, expected {:?}",
            smile, tokens, expect
        );
        assert_eq!(
            token_ids, ids,
            "{} -> {:?}, expected {:?}",
            smile, token_ids, ids
        );
    }

    #[test]
    fn test_tokenize_glyphs() {
        let mut model = GPE::default();
        model.with_vocab_and_merges(
            HashMap::from([
                (model.unk_token.to_owned(), 0),
                ("Co".to_string(), 1),
                ("C".to_string(), 2),
                ("o".to_string(), 3),
                ("[".to_string(), 4),
                ("]".to_string(), 5),
            ]),
            [].to_vec(),
        );
        assert_eq!(model.get_vocab_size(), 6);
        assert_eq!(
            model.tokenize_glyphs("Co").unwrap(),
            [
                Token {
                    id: 2,
                    value: "C".to_string(),
                    offsets: (0, 1)
                },
                Token {
                    id: 3,
                    value: "o".to_string(),
                    offsets: (1, 2)
                },
            ],
        );
        assert_eq!(
            model.tokenize_glyphs("Co🤷").unwrap(),
            [
                Token {
                    id: 2,
                    value: "C".to_string(),
                    offsets: (0, 1)
                },
                Token {
                    id: 3,
                    value: "o".to_string(),
                    offsets: (1, 2)
                },
                Token {
                    id: 0,
                    value: model.unk_token.to_owned(),
                    offsets: (2, 6)
                },
            ],
        );
        assert_eq!(
            model.tokenize_glyphs("[Co🤷]").unwrap(),
            [
                Token {
                    id: 4,
                    value: "[".to_string(),
                    offsets: (0, 1)
                },
                Token {
                    id: 1,
                    value: "Co".to_string(),
                    offsets: (1, 3)
                },
                Token {
                    id: 0,
                    value: model.unk_token.to_owned(),
                    offsets: (3, 7)
                },
                Token {
                    id: 5,
                    value: "]".to_string(),
                    offsets: (7, 8)
                },
            ],
        );
    }

    #[test]
    fn test_untrained() {
        let mut model = GPE::default();
        model.with_vocab_and_merges(
            HashMap::from([
                ("C".to_string(), 0),
                ("o".to_string(), 1),
                ("[".to_string(), 2),
                ("]".to_string(), 3),
                ("Co".to_string(), 4),
                (model.unk_token.to_owned(), 5),
            ]),
            [].to_vec(),
        );
        check_tokenize(&model, "Co", ["C", "o"].to_vec(), [0, 1].to_vec());
        check_tokenize(
            &model,
            "[Co]",
            ["[", "Co", "]"].to_vec(),
            [2, 4, 3].to_vec(),
        );
        check_tokenize(
            &model,
            "[CCo]",
            ["[", "C", "Co", "]"].to_vec(),
            [2, 0, 4, 3].to_vec(),
        );
    }

    #[test]
    fn test_merges() {
        let mut model = GPE::default();
        model.with_vocab_and_merges(
            HashMap::from([
                ("C".to_string(), 0),
                ("o".to_string(), 1),
                ("[".to_string(), 2),
                ("]".to_string(), 3),
                ("Co".to_string(), 4), // Cobalt
                (model.unk_token.to_owned(), 5),
            ]),
            [(0, 1)].to_vec(),
        );
        assert_eq!(model.get_vocab_size(), 7);
        println!("vocab: {:?}", model.vocab.to_owned());
        println!("vocab_r: {:?}", model.vocab_r.to_owned());
        assert_eq!(&model.vocab_r[&6], "Co");
        check_tokenize(&model, "Co", ["Co"].to_vec(), [6].to_vec());
        check_tokenize(
            &model,
            "[Co]",
            ["[", "Co", "]"].to_vec(),
            [2, 4, 3].to_vec(),
        );
    }

    #[test]
    fn serialize() {
        let model = GPE::default();
        let data = serde_json::to_string(&model).unwrap();
        let model_loaded = serde_json::from_str(&data).unwrap();
        assert_eq!(model, model_loaded);
    }
}
