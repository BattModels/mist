use tokenizers::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior, Offsets};
use tokenizers::tokenizer::pattern::Pattern;
use macro_rules_attribute::macro_rules_attribute;
use tokenizers::impl_serde_type;

use regex::Match;
use crate::split::{MATCH_OUTER, MATCH_INNER};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct AtomicComponent;

fn append_split(splits: &mut Vec<(Offsets, bool)>, prev: &mut usize, m: Match, offset: usize) {
    let start = m.start() + offset;
    let end = m.end() + offset;
    if *prev != start {
        splits.push(((*prev, start), false));
    }
    splits.push(((start, end), true));
    *prev = end;
}
impl Pattern for AtomicComponent {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut splits = Vec::with_capacity(inside.len());
        let mut prev = 0;
        for m in MATCH_OUTER.find_iter(inside){
            // Check for Brackets
            if m.as_str().starts_with("[") {
                // Record opening [
                splits.push(((m.start(), m.start()+1), true));
                prev += 1;

                // Record contents between brackets
                for i in MATCH_INNER.find_iter(m.as_str()) {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct SmirkPreTokenizer;

impl PreTokenizer for SmirkPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(AtomicComponent, SplitDelimiterBehavior::Isolated))
    }
}


#[cfg(test)]
mod tests {
    use tokenizers::tokenizer::{OffsetReferential, OffsetType};
    use super::*;

    fn all_matches(smile: &str) -> bool {
        let splits = AtomicComponent.find_matches(smile).unwrap();
        print!("split: {:?}\n", splits);
        splits.into_iter().all(|(_s, m)| m)
    }

    #[test]
    fn check_matches() {
        assert!(all_matches("OC[C@@H]"));
        assert!(all_matches("OC[C@@H][OH]"));
        assert!(all_matches("OC[C@@H][(O)(H)]"));
        assert!(!all_matches("OC[C@@H](O)(H)"));    // Final (H) is not allowed (not organic)
        assert!(all_matches("OC[C@@H](O)([H])"));   // This is fine (In brackets)
        assert!(all_matches("OC[C@@H](O)(C)"));     // This is fine (carbon)
    }

    fn get_split_tokens(tok: SmirkPreTokenizer, smile: &str) -> Vec<String>{
        let mut smile = PreTokenizedString::from(smile);
        tok.pre_tokenize(&mut smile).unwrap();
        smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect()
    }

    #[test]
    fn check_splits() {
        let pretok = SmirkPreTokenizer;
        assert_eq!(get_split_tokens(pretok, "H2O"), ["H", "2", "O"]);
        assert_eq!(get_split_tokens(pretok, "OC[C@@H]"), ["O", "C", "[", "C", "@@", "H", "]"]);
    }

    #[test]
    fn basic() {
        let pretok = SmirkPreTokenizer;
        let mut smile = PreTokenizedString::from("OC[C@@H][OH]");
        pretok.pre_tokenize(&mut smile).unwrap();
        let split: Vec<_> = smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect();
        print!("split: {:?}", split);
    }
}
