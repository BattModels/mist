use std::str::FromStr;

use crate::pre_tokenizers::split_smiles::{BRACKETED, STRUCTURE};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitStructure {
    #[serde_as(as = "DisplayFromStr")]
    boundary: Regex,
}

impl Default for SplitStructure {
    fn default() -> Self {
        SplitStructure {
            boundary: Regex::from_str((STRUCTURE.to_string() + "|" + BRACKETED + r"|\d").as_str())
                .unwrap(),
        }
    }
}

impl PartialEq for SplitStructure {
    fn eq(&self, other: &Self) -> bool {
        self.boundary.as_str() == other.boundary.as_str()
    }
}

impl Pattern for &SplitStructure {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut splits = Vec::with_capacity(inside.len());
        let mut prev: usize = 0;
        for m in self.boundary.find_iter(inside) {
            if prev != m.start() {
                // Mark the content between structural elements as a word
                splits.push(((prev, m.start()), true));
            }

            // Split match into words
            if m.as_str().starts_with("[") {
                // Push the bracket and their contents as 3 words
                splits.push(((m.start(), m.start() + 1), true));
                splits.push(((m.start() + 1, m.end() - 1), true));
                splits.push(((m.end() - 1, m.end()), true));
            } else {
                // Mark the structure as a word
                splits.push(((m.start(), m.end()), true));
            }
            prev = m.end();
        }
        // Handle trailing words
        if prev != inside.len() {
            splits.push(((prev, inside.len()), true));
        }
        Ok(splits)
    }
}

impl PreTokenizer for SplitStructure {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(self, SplitDelimiterBehavior::Isolated))
    }
}

impl SplitStructure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenizer() {
        fn check(pretok: &SplitStructure, smile: &str, expect: &[&str]) {
            let mut smile = PreTokenizedString::from(smile.to_string());
            let _ = pretok.pre_tokenize(&mut smile);
            let split: Vec<String> = smile
                .get_splits(
                    tokenizers::OffsetReferential::Original,
                    tokenizers::OffsetType::Byte,
                )
                .iter()
                .map(|(s, _, _)| s.to_string())
                .collect();
            assert_eq!(split, expect);
        }

        let pretok = SplitStructure::default();
        check(&pretok, "CC", &["CC"]);
        check(&pretok, "C=C", &["C=C"]);
        check(&pretok, "F/C=C/F", &["F", "/", "C=C", "/", "F"]);
        check(&pretok, "CCN(CC)CC", &["CCN", "(", "CC", ")", "CC"]);
        check(&pretok, r"F/C=C\F", &["F", "/", "C=C", r"\", "F"]);
        check(
            &pretok,
            "N[C@@H](C)C(=O)O",
            &[
                "N", "[", "C@@H", "]", "(", "C", ")", "C", "(", "=O", ")", "O",
            ],
        );
        check(
            &pretok,
            "CC1=CC(Br)CCC1",
            &["CC", "1", "=CC", "(", "Br", ")", "CCC", "1"],
        );
        check(
            &pretok,
            "C[NH2+]CCSC[C@@H](O)CO",
            &[
                "C", "[", "NH2+", "]", "CCSC", "[", "C@@H", "]", "(", "O", ")", "CO",
            ],
        );
        check(
            &pretok,
            "[Sc+3].[OH-].[OH-].[OH-]",
            &[
                "[", "Sc+3", "]", ".", "[", "OH-", "]", ".", "[", "OH-", "]", ".", "[", "OH-", "]",
            ],
        );
    }
}
