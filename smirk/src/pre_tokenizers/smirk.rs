use super::split_smiles::{MATCH_INNER, MATCH_OUTER};
use once_cell::sync::Lazy;
use regex::{Match, Regex};
use serde::de::Visitor;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::fmt;
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::tokenizer::{
    Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};

#[derive(Clone)]
pub struct SmirkPreTokenizer {
    outer: Regex,
    inner: Regex,
}

impl SmirkPreTokenizer {
    pub fn new(outer: &str, inner: &str) -> Self {
        Self {
            outer: Regex::new(&outer).unwrap(),
            inner: Regex::new(&inner).unwrap(),
        }
    }
    pub fn split(&self, text: &String) -> Vec<String> {
        self.find_matches(text)
            .unwrap()
            .into_iter()
            .map(|(offset, _)| text.get(offset.0..offset.1).unwrap().to_owned())
            .filter(|tok| !tok.is_empty())
            .collect()
    }
}

impl Default for SmirkPreTokenizer {
    fn default() -> Self {
        SmirkPreTokenizer::new(MATCH_OUTER, MATCH_INNER)
    }
}

impl PartialEq for SmirkPreTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.outer.as_str() == other.outer.as_str() && self.inner.as_str() == other.inner.as_str()
    }
}

impl fmt::Debug for SmirkPreTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmirkPreTokenizer")
            .field("outer", &format_args!("'{}'", &self.outer.as_str()))
            .field("inner", &format_args!("'{}'", &self.inner.as_str()))
            .finish()
    }
}

impl Serialize for SmirkPreTokenizer {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("SmirkPreTokenizer", 2)?;
        state.serialize_field("outer", &self.outer.as_str())?;
        state.serialize_field("inner", &self.inner.as_str())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SmirkPreTokenizer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "SmirkPreTokenizer",
            &["outer", "inner"],
            SmirkPreTokenizerVisitor,
        )
    }
}

struct SmirkPreTokenizerVisitor;
impl<'de> Visitor<'de> for SmirkPreTokenizerVisitor {
    type Value = SmirkPreTokenizer;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "struct SmirkPreTokenizer")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut outer: Option<String> = None;
        let mut inner: Option<String> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "outer" => {
                    if let Some(x) = map.next_value()? {
                        outer = Some(x);
                    }
                }
                "inner" => {
                    if let Some(x) = map.next_value()? {
                        inner = Some(x);
                    }
                }
                _ => {}
            }
        }
        Ok(SmirkPreTokenizer::new(
            outer.expect("Missing `outer`").as_str(),
            inner.expect("Missing `inner`").as_str(),
        ))
    }
}

impl PreTokenizer for SmirkPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(self.to_owned(), SplitDelimiterBehavior::Isolated))
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

impl Pattern for SmirkPreTokenizer {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let mut splits = Vec::with_capacity(inside.len());
        let mut prev = 0;
        let n_inner_groups = self.inner.captures_len();
        static IS_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+$").unwrap());
        for m_outer in self.outer.find_iter(inside) {
            // Check for Brackets
            if m_outer.as_str().starts_with("[") {
                // Record opening [
                splits.push(((m_outer.start(), m_outer.start() + 1), true));
                prev += 1;

                // Record contents between brackets
                let bracketed = &inside[(m_outer.start() + 1)..(m_outer.end() - 1)];
                if let Some(capture) = self.inner.captures(&bracketed) {
                    // Unpack bracketed atoms
                    for i in 1..n_inner_groups {
                        if let Some(m) = capture.get(i) {
                            if IS_NUMBER.is_match(m.as_str()) {
                                // Tokenize numbers as digits
                                for d in m.range() {
                                    let s = d + m_outer.start() + 1;
                                    splits.push(((s, s + 1), true));
                                    prev = s + 1;
                                }
                            } else {
                                append_split(&mut splits, &mut prev, m, m_outer.start() + 1)
                            }
                        }
                    }
                }

                // Check for trailing unmatched characters within the brackets
                if prev != (m_outer.end() - 1) {
                    splits.push(((prev, m_outer.end() - 1), false));
                    prev = m_outer.end() - 1;
                }

                // Record closing [
                assert!(m_outer.as_str().ends_with("]"));
                splits.push(((prev, m_outer.end()), true));
                prev = m_outer.end();
            } else {
                append_split(&mut splits, &mut prev, m_outer, 0);
            }
        }
        // Check for trailing unmatched characters
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false));
        }
        Ok(splits)
    }
}
#[cfg(test)]
pub mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::test_utils::check_serde;
    use tokenizers::tokenizer::{OffsetReferential, OffsetType};

    #[test]
    fn serialize_default() {
        let default = SmirkPreTokenizer::default();
        check_serde(&default);
    }

    #[test]
    fn serialize_pretok() {
        let pretok = SmirkPreTokenizer::new(r".|\[.*?]", ".");
        check_serde(&pretok);
    }

    fn all_matches(tok: &SmirkPreTokenizer, smile: &str) -> bool {
        let splits = tok.find_matches(smile).unwrap();
        print!("split: {:?}\n", splits);
        splits.into_iter().all(|(_s, m)| m)
    }

    fn get_matched_pretokens(tok: &SmirkPreTokenizer, smile: &str) -> Vec<String> {
        tok.find_matches(smile)
            .unwrap()
            .into_iter()
            .filter(|(_, m)| *m)
            .map(|(o, _)| smile[o.0..o.1].into())
            .collect()
    }

    #[test]
    fn check_matches() {
        let pretok = SmirkPreTokenizer::default();
        dbg!(&pretok);
        assert_eq!(
            get_split_tokens(&pretok, "OC[C@@H]"),
            ["O", "C", "[", "C", "@@", "H", "]"]
        );
        assert!(all_matches(&pretok, "OC[C@@H]"));
        assert!(all_matches(&pretok, "OC[C@@H][OH]"));
        assert!(!all_matches(&pretok, "OC[C@@H][(O)(H)]")); // Branches within brackets are invalid
        assert!(!all_matches(&pretok, "OC[C@@H](O)(H)")); // Final (H) is not allowed (not organic)
        assert!(all_matches(&pretok, "OC[C@@H](O)([H])")); // This is fine (In brackets)
        assert!(all_matches(&pretok, "OC[C@@H](O)(C)")); // This is fine (carbon)
        assert!(!all_matches(&pretok, "OC[C@@H22](O)(C)")); // Invalid number of hydrogens
        assert!(all_matches(&pretok, "OC[16C@@](O)(C)")); // Isotope
        assert!(all_matches(&pretok, "Br[Co@OH9](C)(S)(Cl)(F)I")); // Chirality
        assert!(all_matches(&pretok, "Br[Co@OH9](C)(S)(Cl)(F)I")); // Chirality
    }

    fn get_split_tokens(tok: &SmirkPreTokenizer, smile: &str) -> Vec<String> {
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
        let pretok = SmirkPreTokenizer::default();
        let smile = "OC[C@@H]".to_string();
        let split = ["O", "C", "[", "C", "@@", "H", "]"];
        assert_eq!(get_split_tokens(&pretok, smile.as_str()), split);
        assert_eq!(pretok.split(&smile), split);
    }

    #[test]
    fn check_unknown() {
        let pretok = SmirkPreTokenizer::default();
        assert_eq!(get_split_tokens(&pretok, "C🤷"), ["C", "🤷",]);
        assert_eq!(get_split_tokens(&pretok, "🤷"), ["🤷",]);
        assert_eq!(get_split_tokens(&pretok, "🤷C"), ["🤷", "C"]);
        assert_eq!(
            get_split_tokens(&pretok, "C[H🤷]"),
            ["C", "[", "H", "🤷", "]"]
        );
        assert_eq!(get_split_tokens(&pretok, "[🤷]"), ["[", "🤷", "]"]);
        assert_eq!(
            get_split_tokens(&pretok, "[🤷H]C"),
            ["[", "🤷", "H", "]", "C"]
        );
    }

    #[test]
    fn check_digits() {
        let pretok = SmirkPreTokenizer::default();
        dbg!(&pretok);
        // Isotope
        assert_eq!(
            get_split_tokens(&pretok, "[16C]"),
            ["[", "1", "6", "C", "]"]
        );
        //Charge
        assert_eq!(
            get_split_tokens(&pretok, "[C+12]"),
            ["[", "C", "+", "1", "2", "]"]
        );
        // Invalid SMILES (Only 1 digit for hcount)
        assert_eq!(
            get_split_tokens(&pretok, "[CH22+2]"),
            ["[", "C", "H", "2", "2+2", "]"]
        );
        assert_eq!(
            get_matched_pretokens(&pretok, "[CH22+2]"),
            ["[", "C", "H", "2", "]"]
        );
        // hcount + charge
        assert_eq!(
            get_matched_pretokens(&pretok, "[CH2+12]"),
            ["[", "C", "H", "2", "+", "1", "2", "]"]
        );
        // Chirality Permutation
        assert_eq!(
            get_split_tokens(&pretok, "F[As@TB15](Cl)(S)(Br)N"),
            [
                "F", "[", "As", "@TB", "1", "5", "]", "(", "Cl", ")", "(", "S", ")", "(", "Br",
                ")", "N"
            ]
        );
        // Class
        assert_eq!(
            get_split_tokens(&pretok, "[CH4:200]"),
            ["[", "C", "H", "4", ":", "2", "0", "0", "]"]
        );
    }

    #[test]
    fn check_chiral() {
        let pretok = SmirkPreTokenizer::default();
        dbg!(&pretok);
        assert_eq!(get_split_tokens(&pretok, "[C@]"), ["[", "C", "@", "]",],);
        assert_eq!(
            get_split_tokens(&pretok, "[C@H]"),
            ["[", "C", "@", "H", "]",],
        );
        assert_eq!(get_split_tokens(&pretok, "[C@@]"), ["[", "C", "@@", "]",],);
        assert_eq!(
            get_split_tokens(&pretok, "[C@@+2]"),
            ["[", "C", "@@", "+", "2", "]",],
        );
        assert_eq!(
            get_split_tokens(&pretok, "[O@OH2H]"),
            ["[", "O", "@OH", "2", "H", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "[As@TB20]"),
            ["[", "As", "@TB", "2", "0", "]"]
        );
        assert_eq!(
            get_split_tokens(&pretok, "[Xe@SP2]"),
            ["[", "Xe", "@SP", "2", "]"],
        );
        assert_eq!(
            get_split_tokens(&pretok, "[C@AL3]"),
            ["[", "C", "@AL", "3", "]"],
        );
        assert_eq!(
            get_split_tokens(&pretok, "[Fe@TB3+3]"),
            ["[", "Fe", "@TB", "3", "+", "3", "]"],
        );
        assert_eq!(
            get_split_tokens(&pretok, "[*@TH2]"),
            ["[", "*", "@TH", "2", "]",]
        );
    }

    #[test]
    fn basic_smiles() {
        let pretok = SmirkPreTokenizer::default();
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
    fn test_wikipedia_smiles_examples() {
        let pretok = SmirkPreTokenizer::default();
        let examples = [
            "[OH2]",
            "[H]O[H]",
            "[Co+3]",
            "[Na+].[Cl-]",
            "[Ga+]$[As-]",
            "CC-O",
            "O=C=O",
            "C#N",
            "C1CCCC2C1CCCC2",
            "C0CCCCC0C0CCCCC0",
            "C%12",
            "C:1:C:C:C:C:C1",
            "c1ccccc1",
            "n1c[nH]cc1",
            "c1ccccc1-c2ccccc2",
            "COc(cc1)ccc1C#N",
            "FC(Br)(Cl)F",
            "BrC(F)(F)Cl",
            "C(F)(Cl)(F)Br",
            "F/C=C/F",
            r"F/C=C\F",
            "CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C",
            "OC(=O)[C@@H](N)C",
            "C[C@H](N)C(=O)O",
            "CN=C=O",
            "[Cu+2].[O-]S(=O)(=O)[O-]",
            "O=Cc1ccc(O)c(OC)c1COc1cc(C=O)ccc1O",
            "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1",
            "CN1CCC[C@H]1c2cccnc2",
            r"CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\COCCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO",
            r"CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C",
            "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            r"CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C",
            "CC[C@H](O1)CC[C@@]12CCCO2",
            "CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2",
            "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
            "CC(C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
        ];
        dbg!(&pretok);
        for example in examples {
            dbg!(&example);
            assert!(all_matches(&pretok, example));
        }
    }

    #[test]
    fn test_opensmiles_spec() {
        let pretok = SmirkPreTokenizer::default();
        let mut opensmiles_examples = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        opensmiles_examples.push("test");
        opensmiles_examples.push("opensmiles.smi");
        let examples = fs::read_to_string(opensmiles_examples.as_path())
            .expect("failed to open opensmiles.smi");
        for line in examples.lines().filter(|x| !x.starts_with("#")) {
            dbg!(&line);
            assert!(all_matches(&pretok, line));
        }
    }
}
