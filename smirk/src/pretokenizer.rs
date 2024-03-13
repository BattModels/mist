use tokenizers::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior, Offsets};
use tokenizers::tokenizer::pattern::Pattern;

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
                for i in MATCH_INNER.find_iter(m.as_str()) {
                    append_split(&mut splits, &mut prev, i, m.start())
                }
            } else {
                append_split(&mut splits, &mut prev, m, 0);
            }
        }
        Ok(splits)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SmirkPreTokenizer;

impl PreTokenizer for SmirkPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(AtomicComponent, SplitDelimiterBehavior::Isolated))
    }
}


#[cfg(test)]
mod tests {
    use tokenizers::tokenizer::{OffsetType, OffsetReferential};
    use super::*;

    #[test]
    fn basic() {
        let pretok = SmirkPreTokenizer;
        let mut smile = PreTokenizedString::from("OC[C@@H]");
        pretok.pre_tokenize(&mut smile).unwrap();
        let split: Vec<_> = smile
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect();
        print!("split: {:?}", split);

    }
}
