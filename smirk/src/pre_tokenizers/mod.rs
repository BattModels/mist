mod smirk;
mod split_smiles;

use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::SplitDelimiterBehavior;

pub use smirk::SmirkPreTokenizer;

pub fn split_structure() -> Split {
    let pattern = SplitPattern::Regex(r"%|[\(\)]|[/\\]|\[.*?]|\d".to_owned());
    Split::new(pattern, SplitDelimiterBehavior::Isolated, false).unwrap()
}
