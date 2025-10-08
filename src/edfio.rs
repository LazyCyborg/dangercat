use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::io::{Error, ErrorKind};
use std::path::Path;

use edf_reader::sync_reader::SyncEDFReader;
use local_edf_reader::LocalFileReader;
use local_edf_reader::init_sync_reader;

use crate::{RawEEG, EEGInfo, Markers, reference};


pub fn open_file(file_path: &str, raw_eeg: &mut RawEEG) -> std::io::Result<()> {
    match Path::new(file_path).try_exists() {
        Ok(true) => {
            raw_eeg.file_path = Some(file_path.to_string());
            let file = File::open(file_path)?;
            let mut buf_reader = BufReader::new(file);
            let mut header_contents_buffer: [u8; 256] = [0; 256];
            buf_reader.read_exact(&mut header_contents_buffer)?;

            let header_ascii = header_contents_buffer.to_ascii_lowercase();
            let number_of_data_records = &header_contents_buffer[236..243].to_ascii_lowercase();
            let duration_of_recording = &header_contents_buffer[244..251].to_ascii_lowercase();
            let number_of_channels = &header_contents_buffer[252..256].to_ascii_lowercase();
            let head = String::from_utf8_lossy(&header_ascii).to_string();

            println!("HEADER: {:?}\n", head);
            raw_eeg.raw_header = Some(head);
            println!(
                "Number of data records: {:?}\n",
                String::from_utf8_lossy(number_of_data_records)
            );
            println!(
                "Duration of recording: {:?}\n",
                String::from_utf8_lossy(duration_of_recording)
            );
            println!(
                "Number of channels: {:?}\n",
                String::from_utf8_lossy(number_of_channels)
            );
        }
        Ok(false) => {
            let error = Error::from(ErrorKind::InvalidData);
            return Err(error);
        }
        _ => println!("IO operation failed"),
    }

    Ok(())
}

pub fn parse_edf_info_load_data(
    file_path: &str,
    raw_eeg: &mut RawEEG,
    eeg_info: &mut EEGInfo,
    eeg_markers: &mut Markers,
    print_info: bool,
    load_data: bool,
) -> std::io::Result<()> {
    if !Path::new(file_path).try_exists()? {
        return Err(Error::from(ErrorKind::NotFound));
    }

    raw_eeg.file_path = Some(file_path.to_string());

    let edf_reader = init_sync_reader(file_path)?;

    let header = &edf_reader.edf_header;
    raw_eeg.header = Some(header.clone());
    let number_of_channels = header.channels.len();

    if number_of_channels == 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "No channels in EDF file",
        ));
    }
    raw_eeg.number_of_channels = Some(number_of_channels);
    raw_eeg.channels = Some(header.channels.clone());

    eeg_info.num_ch = number_of_channels as i32;
    eeg_info.ch_names = header.channels.iter().map(|c| c.label.clone()).collect();

    let start_time_ms = 0;
    let total_duration_ms = header.number_of_blocks * header.block_duration;

    // Last channel contains the markers
    // this is probably the standard but this might need fixing
    let mut sfreqs = Vec::with_capacity(number_of_channels - 1);

    for channel in &header.channels[..number_of_channels - 1] {
        let sfreq = channel.number_of_samples_in_data_record * 1000 / header.block_duration;
        sfreqs.push(sfreq);
    }

    eeg_info.sfreq = sfreqs[0] as i32;

    match sfreqs.windows(2).all(|w| w[0] == w[1]) {
        true => {raw_eeg.sampling_frequency = Some(sfreqs[0]);
            println!("Sampling rate {:?}", &raw_eeg.sampling_frequency);}
        false => println!(
            "Warning: Channels have different sampling frequencies this is not yet supported"
        ),
    }
    if print_info {
        println!("Number of channels: {number_of_channels}");
        println!("Total duration: {} seconds", total_duration_ms / 1000);
    }
    raw_eeg.total_duration_ms = Some(total_duration_ms);


    if load_data {
        let data = edf_reader.read_data_window(start_time_ms, total_duration_ms)?;
        raw_eeg.edf_data = Some(data.clone());
        println!("Data loaded successfully");

        if data.len() > 1 {
            let eeg_data_only: Vec<Vec<f32>> = data[..data.len()-1].to_vec();

            if let Some(first_len) = eeg_data_only.first().map(|ch| ch.len()) {
                let all_same_length = eeg_data_only.iter().all(|ch| ch.len() == first_len);
                if !all_same_length {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "Channels have different lengths"
                    ));
                }
            }

            raw_eeg.edf_data = Some(eeg_data_only.clone());

            match reference::compute_average_reference_f32(&eeg_data_only) {
                Ok(avg_ref) => {
                    raw_eeg.edf_data_avg_ref = Some(avg_ref);
                }
                Err(e) => {
                    eprintln!("Error computing average reference: {}", e);
                    raw_eeg.edf_data_avg_ref = None;
                }
            }

        } else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Not enough channels in data"
            ));
        }

        if let Some(channel) = header.channels.last() {
            let last_channel_label = &channel.label;
            println!("Last channel label: {}", last_channel_label);

            let last_channel_data = &data[data.len() - 1];

            if last_channel_label.contains("EDF Annotations") {
                parse_edf_annotations(
                    last_channel_data,
                    raw_eeg.sampling_frequency.unwrap_or(1),
                    eeg_markers
                );
                println!("Found {} markers", eeg_markers.n_markers);
            }
        }


    }

    Ok(())
}

pub fn read_edf_data(
    raw_eeg: &mut RawEEG,
    edf_reader: SyncEDFReader<LocalFileReader>,
    start_time_ms: u64,
    total_duration_ms: u64,
) -> std::io::Result<()> {
    let data = edf_reader.read_data_window(start_time_ms, total_duration_ms)?;

    raw_eeg.edf_data = Some(data);

    Ok(())
}


fn parse_edf_annotations(
    signal_data: &[f32],
    sampling_frequency: u64,
    eeg_markers: &mut Markers,
) {
    let mut bytes = Vec::new();
    for &value in signal_data {
        let val = value as u16;
        bytes.push((val & 0xFF) as u8);
        bytes.push((val >> 8) as u8);
    }

    let annotation_str = String::from_utf8_lossy(&bytes);

    let mut first_timestamp: Option<f64> = None;

    for tal in annotation_str.split('\x00') {
        if tal.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = tal.split(|c| c == '\x14' || c == '\x15')
            .filter(|s| !s.is_empty())
            .collect();

        if first_timestamp.is_none() {
            if let Some(onset_str) = parts.first() {
                if let Some(onset_str) = onset_str.strip_prefix('+') {
                    if let Ok(time) = onset_str.trim().parse::<f64>() {
                        first_timestamp = Some(time);
                        println!("First block timestamp offset: {}s", time);
                    }
                }
            }
        }

        if parts.len() < 3 {
            continue;
        }

        if let Some(onset_str) = parts.first() {
            if let Some(onset_str) = onset_str.strip_prefix('+') {
                if let Ok(onset_seconds) = onset_str.trim().parse::<f64>() {
                    // Subtract the first block offset
                    let adjusted_time = onset_seconds - first_timestamp.unwrap_or(0.0);
                    let sample_position = adjusted_time * sampling_frequency as f64;
                    eeg_markers.markers.push(sample_position);
                }
            }
        }
    }

    eeg_markers.n_markers = eeg_markers.markers.len();
}
