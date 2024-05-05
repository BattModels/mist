mod smirk;
mod split_selfies;
mod split_smiles;

use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::SplitDelimiterBehavior;

pub use smirk::SmirkPreTokenizer;

use self::split_smiles::{BRACKETED, STRUCTURE};

pub fn split_structure() -> Split {
    let pattern = SplitPattern::Regex(STRUCTURE.to_string() + "|" + BRACKETED + r"|\d");
    Split::new(pattern, SplitDelimiterBehavior::Isolated, false).unwrap()
}
