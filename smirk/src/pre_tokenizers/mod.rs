mod smirk;
mod split_selfies;
mod split_smiles;

use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{PreTokenizedString, PreTokenizer, Result};

pub use smirk::SmirkPreTokenizer;

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PreTokenizerWrapper {
    PreTokenizer(tokenizers::PreTokenizerWrapper),
    SmirkPreTokenizer(SmirkPreTokenizer),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        match self {
            PreTokenizerWrapper::PreTokenizer(t) => t.pre_tokenize(pretokenized),
            PreTokenizerWrapper::SmirkPreTokenizer(t) => t.pre_tokenize(pretokenized),
        }
    }
}

impl From<SmirkPreTokenizer> for PreTokenizerWrapper {
    fn from(value: SmirkPreTokenizer) -> Self {
        PreTokenizerWrapper::SmirkPreTokenizer(value)
    }
}

impl From<tokenizers::PreTokenizerWrapper> for PreTokenizerWrapper {
    fn from(value: tokenizers::PreTokenizerWrapper) -> Self {
        PreTokenizerWrapper::PreTokenizer(value)
    }
}
