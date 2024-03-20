use std::collections::HashMap;

use pyo3::{pyclass, pymethods, PyResult, Python};
use pyo3::types::{PyAny, PyString};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::normalizers::Strip;
use tokenizers;
use tokenizers::{AddedToken, EncodeInput, OffsetReferential, OffsetType, PaddingDirection, PaddingParams, PaddingStrategy, PostProcessorWrapper, PreTokenizedString, PreTokenizer, TokenizerBuilder, TokenizerImpl};
use dict_derive::{FromPyObject, IntoPyObject};
use crate::pretokenizer::SmirkPreTokenizer;

#[derive(Clone, Debug, FromPyObject, IntoPyObject)]
struct SpecialTokenConfig {
    bos_token: String,
    eos_token: String,
    unk_token: String,
    sep_token: String,
    pad_token: String,
    cls_token: String,
    mask_token: String,
}

impl Default for SpecialTokenConfig {
    fn default() -> Self {
        Self {
            bos_token: "[BOS]".to_string(),
            eos_token: "[EOS]".to_string(),
            unk_token: "[UNK]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            cls_token: "[CLS]".to_string(),
            mask_token: "[MASK]".to_string(),
        }
    }
}

fn as_added_token(token: &str) -> AddedToken {
    AddedToken {
        content: token.to_string(),
        lstrip: true,
        rstrip: true,
        normalized: true,
        special: true,
        single_word: true,
    }
}

impl Into<Vec<AddedToken>> for SpecialTokenConfig {
    fn into(self) -> Vec<AddedToken> {
        [
            as_added_token(&self.bos_token),
            as_added_token(&self.eos_token),
            as_added_token(&self.unk_token),
            as_added_token(&self.sep_token),
            as_added_token(&self.pad_token),
            as_added_token(&self.cls_token),
            as_added_token(&self.mask_token),
        ].to_vec()
    }
}

#[pyclass]
pub struct SmirkTokenizer {
    tokenizer:TokenizerImpl<WordLevel, Strip, SmirkPreTokenizer, PostProcessorWrapper, Fuse>,
    special_tokens:SpecialTokenConfig
}

impl SmirkTokenizer {
    fn from_model(model: WordLevel, special_tokens: Option<SpecialTokenConfig>) -> Self {
        let mut tokenizer = TokenizerBuilder::new()
            .with_model(model)
            .with_pre_tokenizer(Some(SmirkPreTokenizer {is_smiles: is_smiles}))
            .with_normalizer(Some(Strip::new(true, true)))
            .with_decoder(Some(Fuse::new()))
            .with_post_processor(None::<PostProcessorWrapper>)
            .build()
            .unwrap();

        // Add Special Tokens
        let special_tokens = special_tokens.unwrap_or_default().to_owned();
        let added_tokens: Vec<AddedToken> = special_tokens.clone().into();
        tokenizer.add_special_tokens(&added_tokens);
        Self { tokenizer, special_tokens }
    }
}

#[pymethods]
impl SmirkTokenizer {
    #[new]
    #[pyo3(signature=(file=None))]
    fn __new__(file: Option<&str>) -> Self {
        let special_tokens = SpecialTokenConfig::default();
        let model = match file {
            Some(f) => WordLevel::from_file(f, special_tokens.unk_token.to_owned()).unwrap(),
            None => WordLevel::default(),
        };
        SmirkTokenizer::from_model(model, Some(special_tokens))
    }


    #[staticmethod]
    fn from_vocab(file: &str) -> Self {
        let special_tokens = SpecialTokenConfig::default();
        let model = WordLevel::from_file(file, special_tokens.unk_token.to_owned()).unwrap();
        SmirkTokenizer::from_model(model, Some(special_tokens))
    }

    fn set_padding(&mut self){
        let pad_token = &self.special_tokens.pad_token;
        self.tokenizer.add_special_tokens(&[
            AddedToken {
                content: pad_token.to_owned(),
                single_word: true,
                lstrip: true,
                rstrip: true,
                normalized: true,
                special: true
            }
        ]);
        let default_pad = PaddingParams::default();
        let padding = match self.tokenizer.get_padding() {
            Some(pad) => pad,
            None => &default_pad,
        };
        self.tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_token: pad_token.to_owned(),
            pad_id: self.tokenizer.get_vocab(true)[pad_token],
            pad_type_id: padding.pad_type_id,
            pad_to_multiple_of: padding.pad_to_multiple_of,
        }));
    }

    #[getter]
    fn special_tokens(&self) -> SpecialTokenConfig { self.special_tokens.to_owned() }

    fn pretokenize(&self, smile: &PyString) -> PyResult<Vec<String>>{
        let mut pretokenized = PreTokenizedString::from(smile.to_str().unwrap());
        let _ = self.tokenizer.get_pre_tokenizer().unwrap().pre_tokenize(&mut pretokenized);
        let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect::<Vec<String>>()
        ;
        Ok(splits)
    }

    #[pyo3(signature = (smile, add_special_tokens = true))]
    fn encode(&self, smile: &PyString, add_special_tokens: bool) -> PyResult<Encoding>{
        let input = EncodeInput::from(smile.to_str().unwrap());
        let encoding = self.tokenizer.encode_char_offsets(input, add_special_tokens).unwrap();
        Ok(Encoding::from(encoding))
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        Ok(self.tokenizer.decode(&ids, skip_special_tokens).unwrap())
    }

    #[pyo3(signature = (examples, add_special_tokens = true))]
    fn encode_batch(&self, py: Python<'_>, examples: Vec<&PyString>, add_special_tokens: bool) -> PyResult<Vec<Encoding>>{
        let inputs: Vec<EncodeInput> = examples
            .into_iter()
            .map(|x| EncodeInput::from(x.to_str().unwrap()))
            .collect()
        ;
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
    fn decode_batch(&self, py: Python<'_>, ids: Vec<Vec<u32>>, skip_special_tokens: bool) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let sequences = ids.iter().map(|x| &x[..]).collect::<Vec<&[u32]>>();
            Ok(self.tokenizer
                .decode_batch(&sequences, skip_special_tokens)
                .unwrap()
            )
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

    fn __getstate__(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self.tokenizer).unwrap())
    }

    fn __setstate__(&mut self, state: &PyAny) -> PyResult<()> {
        match state.extract::<String>() {
            Ok(s) => {
                self.tokenizer = serde_json::from_str(s.as_str()).unwrap();
                Ok(())
            },
            Err(e) => Err(e),
        }
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }

    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(with_added_tokens)
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
