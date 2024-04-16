use regex::Regex;
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior, Offsets
};

pub struct SplitStructure {
    boundary: Regex,
}

impl Default for SplitStructure {
    fn default() -> Self {
        SplitStructure{
            boundary: Regex::new(r"%\d{2}|[\(\)\./\\]|\[(.*?)\]|\d").unwrap(),
        }
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
            } else if m.as_str().starts_with("[") {
                // Mark the content inside the bracket as a word
                splits.push(((m.start()+1, m.end()-1), true));
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

impl SplitStructure {

}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_tok(tok: &SplitStructure, smile: &str) -> Result<Vec<String>> {
        let mut smile = PreTokenizedString::from(smile);
        tok.pre_tokenize(&mut smile)?;
        let out = smile
            .get_splits(tokenizers::OffsetReferential::Original, tokenizers::OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect();
        Ok(out)
    }

    #[test]
    fn check_pretok() {
        let tok = SplitStructure::default();
        assert_eq!(split_tok(&tok, "[Sc+3].[OH-].[OH-].[OH-]").unwrap(), ["Sc+3", "OH-", "OH-", "OH-"]);
        assert_eq!(split_tok(&tok, "CNC(C)Cc1ccccc1").unwrap(), ["CNC", "C", "Cc", "ccccc"]);
        assert_eq!(split_tok(&tok, "CN1C=NC2=C1C(=O)N(C(=O)N2C)").unwrap(), ["CN", "C=NC", "=C", "C", "=O", "N", "C", "=O", "N", "C"]);
        assert_eq!(split_tok(&tok, "C%12CCCCC%12C").unwrap(), ["C", "CCCCC", "C"]);
    }
}
