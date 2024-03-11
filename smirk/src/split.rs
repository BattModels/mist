use std::vec;

use const_format::concatcp;
use regex::Regex;

// Capture the organic subset + Aromaticity
const ORGANIC_SUBSET: &'static str = r"Cl?|Br?|c|b|[Nn]|[Pp]|[Ss]|[Oo]|I";

// Capture bound symbols
const BONDS: &'static str = r"[\.\-=#\$:/\\]";

// Capture Rings, Branching, and Sterochemistry
const STRUCTURE: &'static str = r"%|[\(\)]|[/\\]|@{1,2}";

// Capture brackets
const BRACKETED: &'static str = r"\[.*\]";

fn outer_bracketed_split(a: &str) -> Vec<String>{
    let re = Regex::new(concatcp!(ORGANIC_SUBSET, "|", BONDS, "|", STRUCTURE, "|", BRACKETED, "|", r"\d")).unwrap();
    let matches: Vec<_> = re.find_iter(a).map(|m| m.as_str().to_string()).collect();
    return matches
}

fn split_chemically_consistent(a: &str) -> Vec<String>{
    let mut tokens = Vec::new();
    for m in outer_bracketed_split(a){
        if m.starts_with("["){
            // drop
        } else {
            tokens.push(m)
        }
    }
    return tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outer(){
        assert_eq!(outer_bracketed_split("CN=C=O"), ["C", "N", "=", "C", "=", "O"]);
    }

    #[test]
    fn chlorine_carbon() {
        assert_eq!(outer_bracketed_split("[2H]ClC"), ["[2H]", "Cl", "C"]);
    }

    #[test]
    fn boron_bromide() {
        assert_eq!(outer_bracketed_split("BrB3"), ["Br", "B", "3"]);
    }


    #[test]
    fn test_brackets(){
        assert_eq!(split_chemically_consistent("[14c]"), ["[", "1", "4", "c", "]"]);
    }

}
