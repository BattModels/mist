mod smirk;
mod split_smiles;

use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::SplitDelimiterBehavior;

pub use smirk::SmirkPreTokenizer;

pub fn split_structure() -> Split {
    let pattern = SplitPattern::Regex(r"\.|%\d{2}|[\(\)]|[/\\]|\[.*?]|\d".to_owned());
    Split::new(pattern, SplitDelimiterBehavior::Isolated, false).unwrap()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use tokenizers::tokenizer::PreTokenizedString;
    use tokenizers::tokenizer::{OffsetReferential, OffsetType};
    use tokenizers::PreTokenizer;

    fn get_splits(pretok: &Split, text: &str) -> Vec<String> {
        let mut pretokenized = PreTokenizedString::from(text);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect()
    }

    #[test]
    fn test_split_structure() {
        let s = split_structure();
        assert_eq!(get_splits(&s, "CC"), ["CC"]);
        assert_eq!(get_splits(&s, "C.C"), ["C", ".", "C"]);
        assert_eq!(get_splits(&s, "C(C)"), ["C", "(", "C", ")"]);
        assert_eq!(get_splits(&s, r"C\C"), ["C", r"\", "C"]);
        assert_eq!(get_splits(&s, "C/C"), ["C", "/", "C"]);
        assert_eq!(get_splits(&s, "C[13C]"), ["C", "[13C]"]);
        assert_eq!(
            get_splits(&s, "C%10ccccc%10C"),
            ["C", "%10", "ccccc", "%10", "C"]
        );
        assert_eq!(get_splits(&s, "C1ccccc1C"), ["C", "1", "ccccc", "1", "C"]);
    }
}
