use crate::{split_selfies, split_smiles};
use macro_rules_attribute::macro_rules_attribute;
use regex::Match;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use tokenizers::{self, impl_serde_type};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PreTokenizerWrapper {
    PreTokenizer(tokenizers::PreTokenizerWrapper),
    SmirkPreTokenizer(SmirkPreTokenizer),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        match self {
            PreTokenizerWrapper::PreTokenizer(t) => t.pre_tokenize(pretokenized),
            PreTokenizerWrapper::SmirkPreTokenizer(t) => t.pre_tokenize(pretokenized),
        }
    }
}

impl From<SmirkPreTokenizer> for PreTokenizerWrapper {
    fn from(value: SmirkPreTokenizer) -> Self {
        PreTokenizerWrapper::SmirkPreTokenizer(value)
    }
}

impl From<tokenizers::PreTokenizerWrapper> for PreTokenizerWrapper {
    fn from(value: tokenizers::PreTokenizerWrapper) -> Self {
        PreTokenizerWrapper::PreTokenizer(value)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct SmirkPreTokenizer {
    is_smiles: bool,
}

impl SmirkPreTokenizer {
    pub fn new(is_smiles: bool) -> Self {
        Self { is_smiles }
    }
}

impl PreTokenizer for SmirkPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(self, SplitDelimiterBehavior::Isolated))
    }
}

fn append_split(splits: &mut Vec<(Offsets, bool)>, prev: &mut usize, m: Match, offset: usize) {
    let start = m.start() + offset;
    let end = m.end() + offset;
    if *prev != start {
        splits.push(((*prev, start), false));
    }
    splits.push(((start, end), true));
    *prev = end;
}

impl Pattern for &SmirkPreTokenizer {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut splits = Vec::with_capacity(inside.len());
        let mut prev = 0;
        let (match_outer, match_inner) = if self.is_smiles {
            (&split_smiles::MATCH_OUTER, &split_smiles::MATCH_INNER)
        } else {
            (&split_selfies::MATCH_OUTER, &split_selfies::MATCH_INNER)
        };
        for m in match_outer.find_iter(inside) {
            // Check for Brackets
            if m.as_str().starts_with("[") {
                // Record opening [
                splits.push(((m.start(), m.start() + 1), true));
                prev += 1;

                // Record contents between brackets
                for i in match_inner.find_iter(m.as_str()) {
                    append_split(&mut splits, &mut prev, i, m.start())
                }

                // Record closing [
                assert!(m.as_str().ends_with("]"));
                splits.push(((prev, m.end()), true));
                prev += 1;
            } else {
                append_split(&mut splits, &mut prev, m, 0);
            }
        }
        Ok(splits)
    }
}
#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use super::*;
    use tokenizers::tokenizer::{OffsetReferential, OffsetType};

    fn all_matches(smile: &str) -> bool {
        let tok = SmirkPreTokenizer { is_smiles: true };
        let splits = tok
            .borrow()
            .find_matches(smile)
            .unwrap();
        print!("split: {:?}\n", splits);
        splits.into_iter().all(|(_s, m)| m)
    }

    #[test]
    fn check_matches() {
        assert!(all_matches("OC[C@@H]"));
        assert!(all_matches("OC[C@@H][OH]"));
        assert!(all_matches("OC[C@@H][(O)(H)]"));
        assert!(!all_matches("OC[C@@H](O)(H)")); // Final (H) is not allowed (not organic)
        assert!(all_matches("OC[C@@H](O)([H])")); // This is fine (In brackets)
        assert!(all_matches("OC[C@@H](O)(C)")); // This is fine (carbon)
    }

    fn get_split_tokens(tok: SmirkPreTokenizer, smile: &str) -> Vec<String> {
        let mut smile = PreTokenizedString::from(smile);
        tok.pre_tokenize(&mut smile).unwrap();
        smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect()
    }

    #[test]
    fn check_smile_splits() {
        let pretok = SmirkPreTokenizer::new(true);
        assert_eq!(
            get_split_tokens(pretok, "OC[C@@H]"),
            ["O", "C", "[", "C", "@@", "H", "]"]
        );
    }

    #[test]
    fn check_selfies_splits() {
        let pretok = SmirkPreTokenizer::new(false);
        assert_eq!(
            get_split_tokens(pretok, "[C][N][=C][=O]"),
            ["[", "C", "]", "[", "N", "]", "[", "=", "C", "]", "[", "=", "O", "]"]
        );
    }

    #[test]
    fn basic_smiles() {
        let pretok = SmirkPreTokenizer::new(true);
        let mut smile = PreTokenizedString::from("OC[C@@H][OH]");
        pretok.pre_tokenize(&mut smile).unwrap();
        let split: Vec<_> = smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect();
        print!("split: {:?}", split);
    }

    #[test]
    fn basic_selfies() {
        let pretok = SmirkPreTokenizer::new(false);
        let mut smile = PreTokenizedString::from("[C][N][=C][=O]");
        pretok.pre_tokenize(&mut smile).unwrap();
        let split: Vec<_> = smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect();
        print!("split: {:?}", split);
    }
}
