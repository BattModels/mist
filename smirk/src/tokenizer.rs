use std::collections::{HashMap, HashSet};

use crate::gpe::GpeTrainer;
use crate::pre_tokenizers::{PreTokenizerWrapper, SmirkPreTokenizer, SplitStructure};
use dict_derive::{FromPyObject, IntoPyObject};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{pyclass, pymethods, PyResult, Python};

use regex::Regex;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::normalizers::Strip;
use tokenizers::{self, DecoderWrapper, ModelWrapper};
use tokenizers::{
    AddedToken, EncodeInput, OffsetReferential, OffsetType, PaddingDirection, PaddingParams,
    PaddingStrategy, PostProcessorWrapper, PreTokenizedString, PreTokenizer, TokenizerBuilder,
    TokenizerImpl,
};

type Tokenizer =
    TokenizerImpl<ModelWrapper, Strip, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;

#[pyclass]
pub struct SmirkTokenizer {
    tokenizer: Tokenizer,
}

impl SmirkTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

#[pymethods]
impl SmirkTokenizer {
    fn __getstate__(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.tokenizer).unwrap())
    }

    #[staticmethod]
    fn __setstate__(state: &str) -> Self {
        let tok = serde_json::from_str(state).unwrap();
        SmirkTokenizer::new(tok)
    }

    #[staticmethod]
    fn from_vocab(file: &str, is_smiles: bool) -> Self {
        let model = WordLevel::from_file(file, "[UNK]".to_string()).unwrap();
        let tokenizer = TokenizerBuilder::new()
            .with_model(model.into())
            .with_pre_tokenizer(Some(SmirkPreTokenizer::new(is_smiles).into()))
            .with_normalizer(Some(Strip::new(true, true)))
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
        self.tokenizer.get_vocab_size(with_added_tokens)
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

    fn no_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }
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
        let norm = self
            .tokenizer
            .get_normalizer()
            .map_or(None, |s| Some(s.to_owned()));
        let decoder = self
            .tokenizer
            .get_decoder()
            .map_or(None, |d| Some(d.to_owned()));

        let mut tokenizer = match self.tokenizer.get_model() {
            ModelWrapper::WordLevel(wl) => {
                let model = BPE::builder()
                    .unk_token(wl.unk_token.to_owned())
                    .build()
                    .unwrap();
                let pt = PreTokenizerWrapper::SplitStructure(SplitStructure::default());
                let tokenizer: Tokenizer = TokenizerBuilder::default()
                    .with_model(ModelWrapper::BPE(model))
                    .with_pre_tokenizer(Some(pt))
                    .with_normalizer(norm)
                    .with_decoder(decoder)
                    .build()
                    .unwrap();
                Ok(tokenizer)
            }
            _ => Err(()),
        }
        .unwrap();

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
                    Some(g.to_owned())
                }
            })
            .collect();

        // Ensure unk token is in the vocabulary
        let unk_token = match tokenizer.get_model() {
            ModelWrapper::BPE(bpe) => bpe.unk_token.to_owned(),
            ModelWrapper::WordLevel(wl) => Some(wl.unk_token.to_owned()),
            _ => None,
        };
        let mut special = Vec::new();
        if let Some(unk_token) = unk_token {
            special.push(AddedToken { content: unk_token, single_word: false, lstrip: false, rstrip: false, normalized: false,  special: true});
        }

        // Configure the trainer
        let mut builder = GpeTrainer::builder();
        builder.alphabet(alphabet);
        builder.special_tokens(special);
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
                    _ => println!("Unknown parameter {:?} ignoring", key),
                }
            }
        }

        // Train tokenizer
        let mut trainer = builder.build().unwrap();
        let trained = py.allow_threads(|| tokenizer.train_from_files(&mut trainer, files).unwrap());
        Ok(SmirkTokenizer::new(trained.to_owned()))
    }
}

#[derive(FromPyObject, IntoPyObject, Debug)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
}

impl From<tokenizers::Encoding> for Encoding {
    fn from(encoding: tokenizers::Encoding) -> Self {
        Self {
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
        }
    }
}
