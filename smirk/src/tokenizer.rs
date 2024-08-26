use std::collections::{HashMap, HashSet};

use crate::gpe::{GpeTrainer, GPE};
use crate::pre_tokenizers::{split_structure, SmirkPreTokenizer};
use crate::wrapper::{ModelWrapper, PreTokenizerWrapper, TrainerWrapper};
use dict_derive::{FromPyObject, IntoPyObject};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{pyclass, pymethods, PyResult, Python};

use regex::Regex;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::{self, normalizers, DecoderWrapper, Model, NormalizerWrapper};
use tokenizers::{
    AddedToken, EncodeInput, OffsetReferential, OffsetType, PaddingDirection, PaddingParams,
    PaddingStrategy, PostProcessorWrapper, PreTokenizedString, PreTokenizer, TokenizerBuilder,
    TokenizerImpl,
};

type Tokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

#[pyclass(dict, module = "smirk.smirk", name = "SmirkTokenizer")]
#[derive(Clone)]
pub struct SmirkTokenizer {
    tokenizer: Tokenizer,
}

impl SmirkTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

fn normalizer() -> normalizers::Sequence {
    let steps: Vec<normalizers::NormalizerWrapper> = [
        normalizers::Replace::new("++", "+2").unwrap().into(),
        normalizers::Replace::new("--", "-2").unwrap().into(),
        normalizers::Strip::new(true, true).into(),
    ]
    .to_vec();
    normalizers::Sequence::new(steps)
}

#[pymethods]
impl SmirkTokenizer {
    #[new]
    fn __new__() -> Self {
        let tokenizer: Tokenizer = TokenizerBuilder::new()
            .with_model(WordLevel::default().into())
            .with_pre_tokenizer(Some(SmirkPreTokenizer::default().into()))
            .with_normalizer(Some(normalizer().into()))
            .with_decoder(Some(Fuse::default().into()))
            .build()
            .unwrap();
        Self { tokenizer }
    }

    fn __getstate__(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.tokenizer).unwrap())
    }

    fn __setstate__(&mut self, state: &str) {
        self.tokenizer = serde_json::from_str(state).unwrap();
    }

    #[staticmethod]
    fn from_vocab(file: &str) -> Self {
        let model = WordLevel::from_file(file, "[UNK]".to_string()).unwrap();
        let tokenizer = TokenizerBuilder::new()
            .with_model(model.into())
            .with_pre_tokenizer(Some(SmirkPreTokenizer::default().into()))
            .with_normalizer(Some(normalizer().into()))
            .with_decoder(Some(Fuse::new().into()))
            .build()
            .unwrap();
        SmirkTokenizer::new(tokenizer)
    }

    fn pretokenize(&self, smile: &PyString) -> PyResult<Vec<String>> {
        let mut pretokenized = PreTokenizedString::from(smile.to_str().unwrap());
        let _ = self
            .tokenizer
            .get_pre_tokenizer()
            .unwrap()
            .pre_tokenize(&mut pretokenized);
        let splits = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect::<Vec<String>>();
        Ok(splits)
    }

    #[pyo3(signature = (smile, add_special_tokens = true))]
    fn encode(&self, smile: &PyString, add_special_tokens: bool) -> PyResult<Encoding> {
        let input = EncodeInput::from(smile.to_str().unwrap());
        let encoding = self
            .tokenizer
            .encode_char_offsets(input, add_special_tokens)
            .unwrap();
        Ok(Encoding::from(encoding))
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        Ok(self.tokenizer.decode(&ids, skip_special_tokens).unwrap())
    }

    #[pyo3(signature = (examples, add_special_tokens = true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        examples: Vec<&PyString>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Encoding>> {
        let inputs: Vec<EncodeInput> = examples
            .into_iter()
            .map(|x| EncodeInput::from(x.to_str().unwrap()))
            .collect();
        // Release the GIL while tokenizing batch
        let out = py.allow_threads(|| {
            self.tokenizer
                .encode_batch_char_offsets(inputs, add_special_tokens)
                .unwrap()
                .into_iter()
                .map(|e| Encoding::from(e))
                .collect()
        });
        Ok(out)
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        ids: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let sequences = ids.iter().map(|x| &x[..]).collect::<Vec<&[u32]>>();
            Ok(self
                .tokenizer
                .decode_batch(&sequences, skip_special_tokens)
                .unwrap())
        })
    }

    #[pyo3(signature = (pretty = false))]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        Ok(self.tokenizer.to_string(pretty).unwrap())
    }

    #[pyo3(signature = (path, pretty = true))]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        self.tokenizer.save(path, pretty).unwrap();
        Ok(())
    }

    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        Ok(Self::new(Tokenizer::from_file(path).unwrap()))
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        let model = self.tokenizer.get_model();
        let mut vocab_size = model.get_vocab_size();
        if with_added_tokens {
            // Count the number of added tokens not already included in the model's vocab
            for (id, _) in self.tokenizer.get_added_tokens_decoder().iter() {
                if model.id_to_token(*id).is_none() {
                    vocab_size += 1;
                }
            }
        }
        vocab_size
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(with_added_tokens)
    }

    fn get_added_tokens_decoder(&self) -> HashMap<u32, String> {
        let mut added: HashMap<u32, String> = HashMap::new();
        for (id, token) in self.tokenizer.get_added_tokens_decoder().iter() {
            added.insert(*id, token.content.to_owned());
        }
        added
    }

    #[pyo3(signature = (input, with_added_tokens=true))]
    fn tokenize(&self, input: String, with_added_tokens: bool) -> Vec<String> {
        self.tokenizer
            .encode(input, with_added_tokens)
            .unwrap()
            .get_tokens()
            .to_vec()
    }

    fn no_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }

    #[pyo3(signature = (**kwargs))]
    fn with_padding(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut params = PaddingParams::default();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract().unwrap();
                match key {
                    "direction" => {
                        let value: &str = value.extract().unwrap();
                        params.direction = match value {
                            "left" => Ok(PaddingDirection::Left),
                            "right" => Ok(PaddingDirection::Right),
                            other => Err(PyValueError::new_err(format!(
                                "Unknown direction {}",
                                other
                            ))),
                        }?
                    }
                    "pad_id" => params.pad_id = value.extract().unwrap(),
                    "pad_type_id" => params.pad_type_id = value.extract().unwrap(),
                    "pad_token" => params.pad_token = value.extract().unwrap(),
                    "length" => {
                        params.strategy = match value.extract().unwrap() {
                            Some(l) => PaddingStrategy::Fixed(l),
                            _ => PaddingStrategy::BatchLongest,
                        }
                    }
                    _ => println!("Unknown kwargs {}, ignoring", key),
                }
            }
        }
        self.tokenizer.with_padding(Some(params));
        Ok(())
    }

    fn add_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| AddedToken {
                content: token.getattr("content").unwrap().to_string(),
                lstrip: token.getattr("lstrip").unwrap().extract().unwrap(),
                rstrip: token.getattr("rstrip").unwrap().extract().unwrap(),
                normalized: token.getattr("normalized").unwrap().extract().unwrap(),
                single_word: token.getattr("single_word").unwrap().extract().unwrap(),
                special: token.getattr("special").unwrap().extract().unwrap(),
            })
            .collect::<Vec<_>>();
        Ok(self.tokenizer.add_tokens(&tokens))
    }

    #[pyo3(signature = (files, **kwargs))]
    fn train(&self, py: Python, files: Vec<String>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        // Construct Trainable Tokenizer
        let model: ModelWrapper = match self.tokenizer.get_model() {
            ModelWrapper::ModelWrapper(mw) => match mw {
                tokenizers::ModelWrapper::WordLevel(wl) => Ok(GPE::from(wl.to_owned())),
                _ => Err(()),
            },
            ModelWrapper::GPE(gpe) => Ok(gpe.to_owned()),
        }
        .unwrap()
        .into();

        // Remove any special tokens (i.e. [PAD]) from the initial vocab
        let is_special = Regex::new(r"\[[A-Z]+?\]").unwrap();
        let alphabet: HashSet<String> = self
            .tokenizer
            .get_vocab(false)
            .keys()
            .filter_map(|g| {
                if is_special.is_match(g) {
                    None
                } else {
                    Some(g.into())
                }
            })
            .collect();

        // Configure the trainer
        let mut builder = GpeTrainer::builder();
        let mut opt_split_structure = false;
        builder.alphabet(alphabet);
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key: &str = key.extract().unwrap();
                match key {
                    "min_frequency" => {
                        builder.min_frequency(value.extract().unwrap());
                    }
                    "vocab_size" => {
                        builder.vocab_size(value.extract().unwrap());
                    }
                    "limit_alphabet" => {
                        builder.limit_alphabet(value.extract().unwrap());
                    }
                    "merge_brackets" => {
                        builder.merge_brackets(value.extract().unwrap());
                    }
                    "split_structure" => {
                        opt_split_structure = value.extract().unwrap();
                    }
                    _ => println!("Unknown parameter {:?} ignoring", key),
                }
            }
        }

        // Build the Smirk-GPE tokenizer
        let mut tok_builder = TokenizerBuilder::default()
            .with_normalizer(self.tokenizer.get_normalizer().cloned())
            .with_model(model)
            .with_decoder(self.tokenizer.get_decoder().cloned());

        if opt_split_structure {
            tok_builder = tok_builder.with_pre_tokenizer(Some(split_structure().into()));
        } else {
            tok_builder = tok_builder.with_pre_tokenizer(None);
        }
        let mut tokenizer: TokenizerImpl<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = tok_builder.build().unwrap();

        // Train tokenizer
        let mut trainer: TrainerWrapper = builder.build().unwrap().into();
        let _ = py.allow_threads(|| tokenizer.train_from_files(&mut trainer, files).unwrap());
        Ok(SmirkTokenizer::new(tokenizer))
    }
}

#[derive(FromPyObject, IntoPyObject, Debug)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
}

impl From<tokenizers::Encoding> for Encoding {
    fn from(encoding: tokenizers::Encoding) -> Self {
        Self {
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
        }
    }
}
