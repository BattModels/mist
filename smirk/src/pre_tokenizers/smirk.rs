use super::{split_selfies, split_smiles};
use macro_rules_attribute::macro_rules_attribute;
use regex::Match;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};
use tokenizers::{self, impl_serde_type};

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

    fn all_matches(tok: SmirkPreTokenizer, smile: &str) -> bool {
        let splits = tok.borrow().find_matches(smile).unwrap();
        print!("split: {:?}\n", splits);
        splits.into_iter().all(|(_s, m)| m)
    }

    #[test]
    fn check_matches() {
        let tok = SmirkPreTokenizer { is_smiles: true };
        assert!(all_matches(tok, "OC[C@@H]"));
        assert!(all_matches(tok, "OC[C@@H][OH]"));
        assert!(all_matches(tok, "OC[C@@H][(O)(H)]"));
        assert!(!all_matches(tok, "OC[C@@H](O)(H)")); // Final (H) is not allowed (not organic)
        assert!(all_matches(tok, "OC[C@@H](O)([H])")); // This is fine (In brackets)
        assert!(all_matches(tok, "OC[C@@H](O)(C)")); // This is fine (carbon)
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
        let pretok = SmirkPreTokenizer::new(true);
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

    #[test]
    fn test_wikipedia_smiles_examples() {
        let tok = SmirkPreTokenizer::new(true);
        all_matches(tok, "N#N");
        all_matches(tok, "CN=C=O");
        all_matches(tok, "[Cu+2].[O-]S(=O)(=O)[O-]");
        all_matches(tok, "O=Cc1ccc(O)c(OC)c1COc1cc(C=O)ccc1O");
        all_matches(
            tok,
            "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1",
        );
        all_matches(tok, "CN1CCC[C@H]1c2cccnc2");
        all_matches(
            tok,
            r"CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\COCCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO",
        );
        all_matches(
            tok,
            r"CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C",
        );
        all_matches(tok, "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5");
        all_matches(tok, "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1");
        all_matches(
            tok,
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
        );
        all_matches(tok, r"CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C");
        all_matches(tok, "CC[C@H](O1)CC[C@@]12CCCO2");
        all_matches(tok, "CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2");
        all_matches(tok, "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N");
    }

    #[test]
    fn test_wikipedia_selfies() {
        let tok = SmirkPreTokenizer::new(false);
        all_matches(tok, "[N][#N]");
        all_matches(tok, "[C][N][=C][=O]");
        all_matches(
            tok,
            "[Cu+2].[O-1][S][=Branch1][C][=O][=Branch1][C][=O][O-1]",
        );
        all_matches(tok, "[O][=C][C][=C][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Ring1][=Branch2][C][O][C][=C][C][Branch1][Ring1][C][=O][=C][C][=C][Ring1][Branch2][O]");
        all_matches(tok, "[C][C][C][=Branch1][C][=C][C][=C][C][=N+1][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][NH1][C][=C][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][C][C][C][=C][N+1][=C][C][=C][C][=C][C][=C][C][=C][Ring1][=Branch1][NH1][C][Ring1][=Branch2][=C][Ring1][=N][C][=C][Ring1][P]");
        all_matches(
            tok,
            "[C][N][C][C][C][C@H1][Ring1][Branch1][C][=C][C][=C][N][=C][Ring1][=Branch1]",
        );
        all_matches(
            tok,
            r"[C][C][C][C@@H1][Branch1][C][O][C][C][\\C][=C][\\C][=C][\\C][#C][C][#C][\\C][=C][\\C][O][C][C][C][C@@H1][Branch1][C][O][C][C][/C][=C][/C][=C][/C][#C][C][#C][/C][=C][/C][O]",
        );
        all_matches(
            tok,
            r"[C][C][=C][Branch2][Ring2][Ring2][C][=Branch1][C][=O][C][C@@H1][Ring1][=Branch1][O][C][=Branch1][C][=O][C@@H1][C@H1][Branch1][Branch2][C][Ring1][Ring1][Branch1][C][C][C][/C][=C][Branch1][C][\\C][/C][=Branch1][C][=O][O][C][C][/C][=C][\\C][=C]",
        );
        all_matches(tok, "[O][C][=C][C@H1][Branch1][Branch1][C@H1][Ring1][Branch1][O][C][=C][Ring1][Ring1][C][=C][Branch1][Ring1][O][C][C][=C][Ring1][Branch2][O][C][=Branch1][C][=O][C][=C][Ring1][#Branch1][C][C][C][=Branch1][C][=O][Ring1][Branch1]");
        all_matches(tok, "[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][Ring1][Branch2]");
        all_matches(tok, "[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][C@@H1][Ring1][#Branch1][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Branch1][C][O][C][=C][Ring1][#Branch2][C][=Branch1][C][=O][O][Ring1][#C]");
        all_matches(
            tok,
            r"[C][C][=Branch1][C][=O][O][C][C][C][Branch1][C][/C][=C][\\C][C@H1][Branch1][=Branch1][C][Branch1][C][C][=C][C][C][C][=C]",
        );
        all_matches(
            tok,
            "[C][C][C@H1][Branch1][C][O][C][C][C@@][Ring1][Ring2][C][C][C][O][Ring1][Branch1]",
        );
        all_matches(tok, "[C][C][Branch1][C][C][C@@][C][C@@H1][Ring1][Ring1][C@@H1][Branch1][C][C][C][=Branch1][C][=O][C][Ring1][Branch2]");
        all_matches(tok, "[O][C][C][C][=C][Branch1][C][C][N+1][=Branch1][Branch1][=C][S][Ring1][=Branch1][C][C][=C][N][=C][Branch1][C][C][N][=C][Ring1][#Branch1][N]");
    }
}
