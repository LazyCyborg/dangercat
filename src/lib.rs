#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::TemplateApp;

use edf_reader::model::EDFChannel;
use edf_reader::model::EDFHeader;
use ndarray::Array2;
use ndarray::Array3;

pub mod edfio;
pub mod signal;
pub mod bvio;
pub mod reference;

#[derive(Debug, Default, Clone)]
pub struct RawEEG {
    pub file_path: Option<String>,
    pub raw_header: Option<String>,
    pub header: Option<EDFHeader>,
    pub number_of_channels: Option<usize>,
    pub channels: Option<Vec<EDFChannel>>,
    pub sampling_frequency: Option<u64>,
    pub total_duration_ms: Option<u64>,
    pub edf_data: Option<Vec<Vec<f32>>>,
    pub bv_data: Option<Vec<Vec<i16>>>,
    pub edf_data_avg_ref: Option<Vec<Vec<f32>>>,
    pub bv_data_avg_ref: Option<Vec<Vec<i16>>>,
}


#[derive(Debug, Default, Clone)]
pub struct EEGInfo {
    pub num_ch: i32,
    pub ch_namesx: Option<Vec<String>>,
    pub ch_names: Vec<String>,
    pub sfreq: i32,
    pub data_orientation: Option<String>,
    pub binary_format: Option<String>,
    pub sampling_interval_in: Option<String>,
    pub sampling_interval: Option<i32>,
}

#[derive(Debug, Default, Clone)]
pub struct Markers {
    pub n_markers: usize,
    pub markers: Vec<f64>
}


#[derive(Debug)]
pub struct EpochsData {
    pub bv_epochs: Array3<i16>,
    pub edf_epochs_data: Array3<f32>,
    pub ch_names: Vec<String>,
    pub tmin: f64,
    pub tmax: f64
}

#[derive(Debug)]
pub struct EvokedData {
    pub evoked: Array2<f64>,
    pub ch_names: Vec<String>,
    pub tmin: f64,
    pub tmax: f64
}
