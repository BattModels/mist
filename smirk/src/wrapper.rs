use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Model, PreTokenizedString, PreTokenizer, Result, Trainer};

use crate::gpe::{GpeTrainer, GPE};
use crate::pre_tokenizers::SmirkPreTokenizer;

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PreTokenizerWrapper {
    PreTokenizer(tokenizers::PreTokenizerWrapper),
    SmirkPreTokenizer(SmirkPreTokenizer),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        match self {
            Self::PreTokenizer(t) => t.pre_tokenize(pretokenized),
            Self::SmirkPreTokenizer(t) => t.pre_tokenize(pretokenized),
        }
    }
}

impl From<SmirkPreTokenizer> for PreTokenizerWrapper {
    fn from(value: SmirkPreTokenizer) -> Self {
        Self::SmirkPreTokenizer(value)
    }
}

impl From<tokenizers::PreTokenizerWrapper> for PreTokenizerWrapper {
    fn from(value: tokenizers::PreTokenizerWrapper) -> Self {
        Self::PreTokenizer(value)
    }
}

impl From<tokenizers::pre_tokenizers::split::Split> for PreTokenizerWrapper {
    fn from(value: tokenizers::pre_tokenizers::split::Split) -> Self {
        Self::from(tokenizers::PreTokenizerWrapper::from(value))
    }
}

impl From<tokenizers::pre_tokenizers::whitespace::Whitespace> for PreTokenizerWrapper {
    fn from(value: tokenizers::pre_tokenizers::whitespace::Whitespace) -> Self {
        Self::from(tokenizers::PreTokenizerWrapper::from(value))
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub enum TrainerWrapper {
    GpeTrainer(GpeTrainer),
    TrainerWrapper(tokenizers::models::TrainerWrapper),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            Self::GpeTrainer(t) => t.should_show_progress(),
            Self::TrainerWrapper(t) => t.should_show_progress(),
        }
    }

    fn train(&self, model: &mut Self::Model) -> Result<Vec<tokenizers::AddedToken>> {
        match self {
            Self::GpeTrainer(t) => match model {
                ModelWrapper::GPE(gpe) => t.train(gpe),
                _ => Err("GPETrainer can only train GPE models".into()),
            },
            Self::TrainerWrapper(t) => match model {
                ModelWrapper::ModelWrapper(model) => t.train(model),
                _ => Err("HuggingFace Trainers can only train HuggingFace Models".into()),
            },
        }
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        match self {
            Self::GpeTrainer(t) => t.feed(iterator, process),
            Self::TrainerWrapper(t) => t.feed(iterator, process),
        }
    }
}

impl From<GpeTrainer> for TrainerWrapper {
    fn from(value: GpeTrainer) -> Self {
        Self::GpeTrainer(value)
    }
}

impl From<tokenizers::models::TrainerWrapper> for TrainerWrapper {
    fn from(value: tokenizers::models::TrainerWrapper) -> Self {
        Self::TrainerWrapper(value)
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum ModelWrapper {
    GPE(GPE),
    ModelWrapper(tokenizers::ModelWrapper),
}

impl Model for ModelWrapper {
    type Trainer = TrainerWrapper;

    fn tokenize(&self, sequence: &str) -> Result<Vec<tokenizers::Token>> {
        match self {
            Self::GPE(t) => t.tokenize(sequence),
            Self::ModelWrapper(t) => t.tokenize(sequence),
        }
    }
    fn get_trainer(&self) -> <Self as Model>::Trainer {
        match self {
            Self::GPE(t) => t.get_trainer().into(),
            Self::ModelWrapper(t) => t.get_trainer().into(),
        }
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Self::GPE(t) => t.id_to_token(id),
            Self::ModelWrapper(t) => t.id_to_token(id),
        }
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::GPE(t) => t.token_to_id(token),
            Self::ModelWrapper(t) => t.token_to_id(token),
        }
    }
    fn get_vocab_size(&self) -> usize {
        match self {
            Self::GPE(t) => t.get_vocab_size(),
            Self::ModelWrapper(t) => t.get_vocab_size(),
        }
    }
    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        match self {
            Self::GPE(t) => t.get_vocab(),
            Self::ModelWrapper(t) => t.get_vocab(),
        }
    }
    fn save(
        &self,
        folder: &std::path::Path,
        prefix: Option<&str>,
    ) -> Result<Vec<std::path::PathBuf>> {
        match self {
            Self::GPE(t) => t.save(folder, prefix),
            Self::ModelWrapper(t) => t.save(folder, prefix),
        }
    }
}

impl From<GPE> for ModelWrapper {
    fn from(value: GPE) -> Self {
        Self::GPE(value)
    }
}
impl From<tokenizers::ModelWrapper> for ModelWrapper {
    fn from(value: tokenizers::ModelWrapper) -> Self {
        Self::ModelWrapper(value)
    }
}

impl From<tokenizers::models::wordlevel::WordLevel> for ModelWrapper {
    fn from(value: tokenizers::models::wordlevel::WordLevel) -> Self {
        Self::from(tokenizers::ModelWrapper::from(value))
    }
}

#[cfg(test)]
mod test {
    use crate::pre_tokenizers::split_structure;
    use crate::test_utils::check_serde;
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    #[test]
    fn serialize_models() {
        let model = (
            ModelWrapper::GPE(GPE::default()),
            ModelWrapper::ModelWrapper(WordLevel::default().into()),
        );
        check_serde(&model.0.clone());
        check_serde(&model.1.clone());
        check_serde(&model);
    }

    #[test]
    fn serialize_pretok() {
        let pretok = (
            PreTokenizerWrapper::SmirkPreTokenizer(SmirkPreTokenizer::default()),
            PreTokenizerWrapper::PreTokenizer(split_structure().into()),
        );
        check_serde(&pretok.0.clone());
        check_serde(&pretok);
    }
}
