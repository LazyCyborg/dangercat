use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use egui::Key;
use egui::Vec2;
use egui::Color32;
use egui_file_dialog::FileDialog;
use egui_plot::{Text, Line, Plot, PlotPoint, VLine};

use ndarray::Array2;

use crate::{RawEEG, EEGInfo,Markers, edfio, bvio, signal};

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Clone, Copy, Debug)]
enum DataFormat {
    EDF,
    BrainVision,
}

enum ProcessedDataType {
    BV(Array2<i16>),
    EDF(Array2<f32>),
}


#[derive(serde::Deserialize, serde::Serialize, PartialEq, Clone, Copy, Debug)]
enum ReferenceType {
    Original,
    AverageReference,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    data_format: DataFormat,
    #[serde(skip)]
    file_dialog: FileDialog,
    edf_file: Option<PathBuf>,
    #[serde(skip)]
    raw_eeg: RawEEG,
    #[serde(skip)]
    eeg_info: EEGInfo,
    #[serde(skip)]
    eeg_markers: Markers,
    #[serde(skip)]
    loading_receiver: Option<Receiver<Result<(RawEEG, EEGInfo, Markers), std::io::Error>>>,
    #[serde(skip)]
    artifact_receiver: Option<Receiver<Result<ProcessedDataType, std::io::Error>>>,
    #[serde(skip)]
    filtering_receiver: Option<Receiver<Result<ProcessedDataType, std::io::Error>>>,
    #[serde(skip)]
    show_data: bool,
    apply_notch_filter: bool,
    selected_channel: usize,
    reference_type: ReferenceType,
    unselected_channels: Vec<usize>,
    decimation_factor: usize,
    x_view: f64,
    y_view_min: f64,
    y_view_max: f64,
    plot_zoom_factor: Vec2,
    gain: f64,
    tmin_cut: f64,
    tmax_cut: f64,
    lfreq: f64,
    hfreq: f64,
    channel_colors: Vec<Color32>,
    global_color: Color32,
    selected_channel_for_color: usize,
    ruler_position: Option<(f64, f64)>,  // (x, y) position of the ruler
    ruler_width: f64,                    // Width of the ruler in seconds
    ruler_height: f64,                   // Height of the ruler in microvolts
    ruler_dragging: bool,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            data_format: DataFormat::EDF,
            file_dialog: FileDialog::new(),
            edf_file: None,
            raw_eeg: RawEEG::default(),
            eeg_info: EEGInfo::default(),
            eeg_markers: Markers::default(),
            loading_receiver: None,
            filtering_receiver: None,
            apply_notch_filter: false,
            artifact_receiver: None,
            show_data: false,
            reference_type: ReferenceType::Original,
            selected_channel_for_color: 0,
            global_color: Color32::WHITE,
            selected_channel: 0,
            decimation_factor: 100,
            x_view: 0.0,
            y_view_min: 0.0,
            y_view_max: 600.0,
            gain: 1.0,
            plot_zoom_factor: Vec2::new(1.0, 1.0),
            unselected_channels: Vec::new(),
            channel_colors: Vec::new(),
            tmin_cut: 0.002,
            tmax_cut: 0.005,
            lfreq: 1.0,
            hfreq: 45.0,
            ruler_position: None,
            ruler_width: 1.0,  // Default width: 1 second
            ruler_height: 50.0, // Default height: 50 microvolts
            ruler_dragging: false,
        }
    }
}

impl TemplateApp {
    fn min_max_decimate<T>(
        &self,
        data: &[T],
        start_sample: usize,
        decimation: usize,
        offset: f64,
        sampling_frequency: f64
    ) -> Vec<[f64; 2]>
    where
        T: Copy + PartialOrd + Into<f64>,
    {
        if decimation <= 1 {
            return data.iter().enumerate().map(|(i, &sample)| {
                let x = (start_sample + i) as f64 / sampling_frequency;
                let y = (sample.into() / 100.0) * self.gain + offset;
                [x, y]
            }).collect();
        }

        let mut points = Vec::new();
        for chunk in data.chunks(decimation) {
            let chunk_start = (points.len() / 2) * decimation;
            let time_base = (start_sample + chunk_start) as f64 / sampling_frequency;

            if let Some(&first) = chunk.first() {
                let (min_val, max_val) = chunk.iter().fold((first, first), |(min, max), &val| {
                    (if val < min { val } else { min }, if val > max { val } else { max })
                });

                points.push([time_base, (min_val.into() / 100.0) * self.gain + offset]);
                points.push([time_base + (decimation as f64 * 0.5) / sampling_frequency,
                            (max_val.into() / 100.0) * self.gain + offset]);
            }
        }
        points
    }
}

impl TemplateApp {

    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        }
    }

}

impl eframe::App for TemplateApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        if let Some(receiver) = &self.loading_receiver {
            match receiver.try_recv() {
                Ok(Ok((new_raw_eeg, new_eeg_info, new_markers))) => {
                    self.raw_eeg = new_raw_eeg;
                    self.eeg_info = new_eeg_info;
                    self.eeg_markers = new_markers;
                    self.loading_receiver = None;
                    if let Some(ref data_vec) = self.raw_eeg.edf_data {
                        self.channel_colors = vec![Color32::WHITE; data_vec.len()];
                    } else if let Some(ref data_vec) = self.raw_eeg.bv_data {
                        self.channel_colors = vec![Color32::WHITE; data_vec.len()];
                    }

                }
                Ok(Err(e)) => {
                    eprintln!("Error loading EDF: {}", e);
                    self.loading_receiver = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still loading - could show a spinner here
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Loading thread disconnected");
                    self.loading_receiver = None;
                }
            }
        }

        if let Some(receiver) = &self.filtering_receiver {
            match receiver.try_recv() {
                Ok(Ok(processed_data)) => {
                    match processed_data {
                        ProcessedDataType::EDF(data) => {
                            let data_vec = data.outer_iter().map(|row| row.to_vec()).collect();
                            self.raw_eeg.edf_data = Some(data_vec);
                        }
                        ProcessedDataType::BV(data) => {
                            let data_vec = data.outer_iter().map(|row| row.to_vec()).collect();
                            self.raw_eeg.bv_data = Some(data_vec);
                        }
                    }
                    self.filtering_receiver = None;
                }
                Ok(Err(e)) => {
                    eprintln!("Error filtering data: {}", e);
                    self.filtering_receiver = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still filtering
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Filtering thread disconnected");
                    self.filtering_receiver = None;
                }
            }
        }

        if let Some(receiver) = &self.artifact_receiver {
            match receiver.try_recv() {
                Ok(Ok(processed_data)) => {
                    match processed_data {
                        ProcessedDataType::BV(data) => {
                            let data_vec = data.outer_iter().map(|row| row.to_vec()).collect();
                            self.raw_eeg.bv_data = Some(data_vec);
                            self.raw_eeg.edf_data = None;
                        }
                        ProcessedDataType::EDF(data) => {
                            let data_vec = data.outer_iter().map(|row| row.to_vec()).collect();
                            self.raw_eeg.edf_data = Some(data_vec);
                            self.raw_eeg.bv_data = None;
                        }
                    }
                    self.artifact_receiver = None;
                }
                Ok(Err(e)) => {
                    eprintln!("Error processing artifact: {}", e);
                    self.artifact_receiver = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Artifact processing thread disconnected");
                    self.artifact_receiver = None;
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::MenuBar::new().ui(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("Dangercat EEG reader");

            egui::ComboBox::from_label("Data format")
                .selected_text(format!("{:?}", self.data_format))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.data_format, DataFormat::EDF, "EDF");
                    ui.selectable_value(&mut self.data_format, DataFormat::BrainVision, "BrainVision");
                });

            if ui.button("Pick EEG file").clicked() {
                self.file_dialog.pick_file();
            }
            let default_string = PathBuf::new();
            ui.label(format!("Picked file: {:?}", self.edf_file.as_ref().unwrap_or(&default_string)));
            self.file_dialog.update(ctx);

            // Check if the user picked a file.
            if let Some(path) = self.file_dialog.take_picked() {
                self.edf_file = Some(path.to_path_buf());
            }

            ui.separator();

            if ui.button("Read header").clicked() {
                match &self.edf_file {
                    Some(path) => {
                        let data_format = self.data_format;
                        std::thread::spawn({
                            let path = path.clone();
                            let mut raw_eeg = std::mem::take(&mut self.raw_eeg);
                            let mut eeg_info = std::mem::take(&mut self.eeg_info);
                            let mut eeg_markers = std::mem::take(&mut self.eeg_markers);
                            move || {
                                match path.to_str() {
                                    Some(path_str) => {
                                        match data_format {
                                            DataFormat::EDF => {
                                                let _ = edfio::parse_edf_info_load_data(
                                                    path_str, &mut raw_eeg,
                                                    &mut eeg_info, &mut eeg_markers, true, false
                                                );
                                            }
                                            DataFormat::BrainVision => {
                                                if let Ok(header) = bvio::get_header(&Some(path_str.to_string())) {
                                                    if let Ok(info) = bvio::parse_header(&header) {
                                                        println!("BrainVision header parsed: {:?}", info);

                                                    }
                                                }
                                            }
                                        }
                                    }
                                    None => {
                                        eprint!("Error reading file")
                                    }
                                }
                            }
                        });
                    }
                    None => {
                        ui.label("No file selected");
                    }
                }
            }

            ui.separator();

            if ui.button("Load file into memory").clicked() {
                match &self.edf_file {
                    Some(path) => {
                        let (sender, receiver) = std::sync::mpsc::channel();
                        self.loading_receiver = Some(receiver);
                        let data_format = self.data_format;

                        std::thread::spawn({
                            let path = path.clone();
                            let sender = sender;
                            move || {
                                let mut raw_eeg = RawEEG::default();
                                let mut eeg_info = EEGInfo::default();
                                let mut eeg_markers = Markers::default();

                                match path.to_str() {
                                    Some(path_str) => {
                                        let result = match data_format {
                                            DataFormat::EDF => {
                                                edfio::parse_edf_info_load_data(
                                                    path_str, &mut raw_eeg,
                                                    &mut eeg_info, &mut eeg_markers, false, true
                                                )
                                            }
                                            DataFormat::BrainVision => {

                                                bvio::load_bv_data(path_str, &mut raw_eeg, &mut eeg_info, &mut eeg_markers)

                                            }
                                        };

                                        match result {
                                            Ok(_) => { let _ = sender.send(Ok((raw_eeg, eeg_info, eeg_markers))); }
                                            Err(e) => { let _ = sender.send(Err(e)); }
                                        }
                                    }
                                    None => {
                                        let _ = sender.send(Err(std::io::Error::new(
                                            std::io::ErrorKind::InvalidData,
                                            "Invalid file path"
                                        )));
                                    }
                                }
                            }
                        });
                    }
                    None => {
                        ui.label("No file selected");
                    }
                }
            }

            if self.loading_receiver.is_some() {
                match self.data_format {
                    DataFormat::EDF => {
                        ui.label("Loading EDF data...");
                        ui.spinner();
                    }
                    DataFormat::BrainVision => {
                        ui.label("Loading BrainVision data...");
                        ui.spinner();
                    }
                }

            }

            if ui.button("Plot EEG").clicked() {self.show_data = true;}
            ui.separator();
            ui.heading("Filter settings");
            ui.checkbox(&mut self.apply_notch_filter, "Apply 50 Hz notch filter");
            egui::ComboBox::from_label("Highpass filter lfreq")
                .selected_text(format!("{:?}", self.lfreq))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.lfreq, 0.1, "0.1");
                    ui.selectable_value(&mut self.lfreq, 0.2, "0.2");
                    ui.selectable_value(&mut self.lfreq, 0.3, "0.3");
                    ui.selectable_value(&mut self.lfreq, 0.5, "0.5");
                    ui.selectable_value(&mut self.lfreq, 1.0, "1.0");
                    ui.selectable_value(&mut self.lfreq, 2.0, "2.0");
                }
            );

            egui::ComboBox::from_label("Lowpass filter hfreq")
                .selected_text(format!("{:?}", self.hfreq))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.hfreq, 30.0, "30");
                    ui.selectable_value(&mut self.hfreq, 40.0, "40");
                    ui.selectable_value(&mut self.hfreq, 45.0, "45");
                    ui.selectable_value(&mut self.hfreq, 50.0, "50");
                    ui.selectable_value(&mut self.hfreq, 70.0, "70");
                    ui.selectable_value(&mut self.hfreq, 100.0, "100");
                }
            );


            if ui.button("Filter data").clicked() {
                let (sender, receiver) = std::sync::mpsc::channel();
                self.filtering_receiver = Some(receiver);
                let info = self.eeg_info.clone();
                let lfreq = self.lfreq;
                let hfreq = self.hfreq;
                let apply_notch = self.apply_notch_filter;

                match self.data_format {
                    DataFormat::EDF => {
                        if let Some(data_vec) = self.raw_eeg.edf_data.clone() {
                            std::thread::spawn(move || {
                                let data = signal::vec_to_ndarray(&data_vec);
                                let result = signal::edf_hp_filter(lfreq, &info, &data)
                                    .and_then(|filtered| signal::edf_lp_filter(hfreq, &info, &filtered))
                                    .and_then(|filtered| {
                                        if apply_notch {
                                            signal::edf_notch_filter_50hz(&info, &filtered)
                                        } else {
                                            Ok(filtered)
                                        }
                                    })
                                    .map(ProcessedDataType::EDF);
                                let _ = sender.send(result.map_err(|e| std::io::Error::new(
                                    std::io::ErrorKind::Other, e.to_string()
                                )));
                            });
                        }
                    }
                    DataFormat::BrainVision => {
                        if let Some(data_vec) = self.raw_eeg.bv_data.clone() {
                            std::thread::spawn(move || {
                                let data = signal::vec_to_ndarray(&data_vec);
                                let result = signal::hp_filter(lfreq, &info, &data)
                                    .and_then(|filtered| signal::lp_filter(hfreq, &info, &filtered))
                                    .and_then(|filtered| {
                                        if apply_notch {
                                            signal::notch_filter_50hz(&info, &filtered)
                                        } else {
                                            Ok(filtered)
                                        }
                                    })
                                    .map(ProcessedDataType::BV);
                                let _ = sender.send(result.map_err(|e| std::io::Error::new(
                                    std::io::ErrorKind::Other, e.to_string()
                                )));
                            });
                        }
                    }
                }
            }


            if self.filtering_receiver.is_some() {
                ui.label("Filtering data...");
                ui.spinner();
            }

            if self.show_data {
                if !self.eeg_info.ch_names.is_empty() && self.eeg_info.sfreq > 0 {
                    // Keyboard controls
                    if ctx.input(|i|i.key_pressed(Key::K)){
                        self.y_view_max += 10.0;
                        self.y_view_min += 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::J)){
                        self.y_view_max -= 10.0;
                        self.y_view_min -= 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::ArrowRight)){
                        self.x_view += 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::ArrowLeft)){
                        self.x_view -= 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::L)){
                        self.x_view += 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::H)){
                        self.x_view -= 10.0
                    }
                    if ctx.input(|i|i.key_pressed(Key::ArrowUp)){
                        self.gain *= 1.1;
                    }
                    if ctx.input(|i|i.key_pressed(Key::ArrowDown)){
                        self.gain /= 1.1;
                    }


                    Plot::new("my_plot")
                        .show_x(true)
                        .show_y(false)
                        .show(ui, |plot_ui| {
                            let sampling_frequency = self.eeg_info.sfreq as f64;
                            let channel_names = &self.eeg_info.ch_names;

                            let start_time = self.x_view;
                            let end_time = self.x_view + 10.0;
                            let start_sample = ((start_time * sampling_frequency) as usize).max(0);
                            let end_sample = (end_time * sampling_frequency) as usize;

                            let mut offset = 0.0;
                            let channel_offset = 10.0;

                            match self.data_format {
                                DataFormat::EDF => {
                                    let data_vec = match self.reference_type {
                                        ReferenceType::Original => &self.raw_eeg.edf_data,
                                        ReferenceType::AverageReference => &self.raw_eeg.edf_data_avg_ref,
                                    };
                                    if let Some(data_vec) = data_vec {
                                        for ch in 0..data_vec.len() {
                                            if !self.unselected_channels.contains(&ch) {
                                                let channel_slice = &data_vec[ch];
                                                if start_sample < channel_slice.len() {
                                                    let actual_end = end_sample.min(channel_slice.len());
                                                    let visible_data = &channel_slice[start_sample..actual_end];
                                                    let points = self.min_max_decimate(visible_data, start_sample, self.decimation_factor, offset, sampling_frequency);
                                                    let line_color = self.channel_colors[ch];
                                                    plot_ui.line(Line::new(format!("ch_{}", ch), points).color(line_color));
                                                    let text_point = PlotPoint::new(self.x_view + 0.1, offset);
                                                    plot_ui.text(Text::new(
                                                        channel_names[ch].clone(),
                                                        text_point,
                                                        channel_names[ch].clone(),
                                                    ));
                                                    offset += channel_offset;
                                                }
                                            }
                                        }
                                    }
                                }
                                DataFormat::BrainVision => {
                                    let data_vec = match self.reference_type {
                                        ReferenceType::Original => &self.raw_eeg.bv_data,
                                        ReferenceType::AverageReference => &self.raw_eeg.bv_data_avg_ref,
                                    };
                                    if let Some(data_vec) = data_vec {
                                        for ch in 0..data_vec.len() {
                                            if !self.unselected_channels.contains(&ch) {
                                                let channel_slice = &data_vec[ch];
                                                if start_sample < channel_slice.len() {
                                                    let actual_end = end_sample.min(channel_slice.len());
                                                    let visible_data = &channel_slice[start_sample..actual_end];
                                                    let points = self.min_max_decimate(visible_data, start_sample, self.decimation_factor, offset, sampling_frequency);
                                                    let line_color = self.channel_colors[ch];
                                                    plot_ui.line(Line::new(format!("ch_{}", ch), points).color(line_color));
                                                    let text_point = PlotPoint::new(self.x_view + 0.1, offset);
                                                    plot_ui.text(Text::new(
                                                        channel_names[ch].clone(),
                                                        text_point,
                                                        channel_names[ch].clone(),
                                                    ));
                                                    offset += channel_offset;
                                                }
                                            }
                                        }
                                    }
                                }
                            }


                            let visible_channels = self.eeg_info.num_ch as usize - self.unselected_channels.len();
                            let total_height = visible_channels as f64 * channel_offset;
                            plot_ui.set_plot_bounds_y(-channel_offset..=(total_height + channel_offset));
                            plot_ui.set_plot_bounds_x(self.x_view..=(self.x_view + 10.0));
                            for marker_pos in &self.eeg_markers.markers {
                                let marker_time = *marker_pos / sampling_frequency;
                                plot_ui.vline(VLine::new("Annotation", marker_time));
                            }
                            if let Some(ruler_pos_val) = self.ruler_position {
                                let mut ruler_pos = ruler_pos_val;
                                let plot_ruler_height = (self.ruler_height / 100.0) * self.gain;

                                let ruler_rect = egui::Rect::from_min_size(
                                    egui::pos2(ruler_pos.0 as f32, ruler_pos.1 as f32),
                                    egui::vec2(self.ruler_width as f32, plot_ruler_height as f32),
                                );
                                if let Some(pointer) = plot_ui.pointer_coordinate() {
                                    if plot_ui.ctx().input(|i| i.pointer.primary_pressed()) {
                                        if ruler_rect.contains(egui::pos2(pointer.x as f32, pointer.y as f32)) {
                                            self.ruler_dragging = true;
                                        }
                                    }
                                }
                                if self.ruler_dragging && plot_ui.ctx().input(|i| i.pointer.primary_down()) {
                                    let drag_delta = plot_ui.pointer_coordinate_drag_delta();
                                    ruler_pos.0 += drag_delta.x as f64;
                                    ruler_pos.1 += drag_delta.y as f64;
                                    self.ruler_position = Some(ruler_pos);
                                }

                                if plot_ui.ctx().input(|i| i.pointer.primary_released()) {
                                    self.ruler_dragging = false;
                                }
                                let color = egui::Color32::from_rgba_unmultiplied(100, 150, 255, 180);
                                let v_points = vec![[ruler_pos.0, ruler_pos.1], [ruler_pos.0, ruler_pos.1 + plot_ruler_height]];
                                plot_ui.line(Line::new("ruler_v".to_string(), v_points).color(color).width(2.0));
                                let h_points = vec![[ruler_pos.0, ruler_pos.1], [ruler_pos.0 + self.ruler_width, ruler_pos.1]];
                                plot_ui.line(Line::new("ruler_h".to_string(), h_points).color(color).width(2.0));
                                let height_text = format!("{:.0} µV", self.ruler_height);
                                let text_pos_v = PlotPoint::new(ruler_pos.0, ruler_pos.1 + plot_ruler_height / 2.0);
                                plot_ui.text(Text::new("ruler_text_h".to_string(), text_pos_v, height_text).color(color));
                                let width_text = format!("{:.2} s", self.ruler_width);
                                let text_pos_h = PlotPoint::new(ruler_pos.0 + self.ruler_width / 2.0, ruler_pos.1);
                                plot_ui.text(Text::new("ruler_text_w".to_string(), text_pos_h, width_text).color(color));
                            }
                        });

                } else {
                    ui.label("No data available to plot");
                }
            }


            ui.separator();
            ui.separator();

            // ui.add(egui::github_link_file!(
            //     "https://github.com/emilk/eframe_template/blob/main/",
            //     "Source code."
            // ));

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
        });

        egui::SidePanel::right("analysis_panel")
            .min_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Artefact removal");

                ui.add(egui::Slider::new(&mut self.tmin_cut, 0.001..=0.020)
                    .text("Pre-stimulus (s)")
                    .suffix(" s"));
                ui.add(egui::Slider::new(&mut self.tmax_cut, 0.001..=0.050)
                    .text("Post-stimulus (s)")
                    .suffix(" s"));

                ui.separator();


                if ui.button("Remove TMS pulse (zero)").clicked() {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    self.artifact_receiver = Some(receiver);

                    let info = self.eeg_info.clone();
                    let markers = self.eeg_markers.clone();
                    let tmin = self.tmin_cut;
                    let tmax = self.tmax_cut;

                    match self.data_format {
                        DataFormat::EDF => {
                            if let Some(data_vec) = self.raw_eeg.edf_data.clone() {
                                std::thread::spawn(move || {
                                    let data = signal::vec_to_ndarray(&data_vec);
                                    let result = signal::remove_tms_pulse_f32(tmin, tmax, &markers, &info, &data)
                                        .map(ProcessedDataType::EDF);
                                    let _ = sender.send(result.map_err(|e|
                                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                                    ));
                                });
                            }
                        }
                        DataFormat::BrainVision => {
                            if let Some(data_vec) = self.raw_eeg.bv_data.clone() {
                                std::thread::spawn(move || {
                                    let data = signal::vec_to_ndarray(&data_vec);
                                    let result = signal::remove_tms_pulse(tmin, tmax, &markers, &info, &data)
                                        .map(ProcessedDataType::BV);
                                    let _ = sender.send(result.map_err(|e|
                                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                                    ));
                                });
                            }
                        }
                    }
                }

                if ui.button("Remove and interpolate pulse").clicked() {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    self.artifact_receiver = Some(receiver);

                    let info = self.eeg_info.clone();
                    let markers = self.eeg_markers.clone();
                    let tmin = self.tmin_cut;
                    let tmax = self.tmax_cut;

                    match self.data_format {
                        DataFormat::EDF => {
                            if let Some(data_vec) = self.raw_eeg.edf_data.clone() {
                                std::thread::spawn(move || {
                                    let data = signal::vec_to_ndarray(&data_vec);
                                    let result = signal::rm_interp_tms_pulse_f32(tmin, tmax, &markers, &info, &data)
                                        .map(ProcessedDataType::EDF);
                                    let _ = sender.send(result.map_err(|e|
                                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                                    ));
                                });
                            }
                        }
                        DataFormat::BrainVision => {
                            if let Some(data_vec) = self.raw_eeg.bv_data.clone() {
                                std::thread::spawn(move || {
                                    let data = signal::vec_to_ndarray(&data_vec);
                                    let result = signal::rm_interp_tms_pulse(tmin, tmax, &markers, &info, &data)
                                        .map(ProcessedDataType::BV);
                                    let _ = sender.send(result.map_err(|e|
                                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                                    ));
                                });
                            }
                        }
                    }
                }

                if self.artifact_receiver.is_some() {
                    ui.label("Processing artefact...");
                    ui.spinner();
                }

                ui.separator();
                ui.heading("Plot tools");
                ui.add(egui::Slider::new(&mut self.decimation_factor, 1..=500)
                    .text("Decimation factor for plotting")
                    );
                ui.separator();

                egui::ComboBox::from_label("Reference")
                    .selected_text(format!("{:?}", self.reference_type))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.reference_type, ReferenceType::Original, "As recorded");
                        ui.selectable_value(&mut self.reference_type, ReferenceType::AverageReference, "Average reference");
                    });


                ui.separator();
                ui.heading("Channel Colors");

                ui.label("Global color for all channels:");
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut self.global_color,
                    egui::color_picker::Alpha::Opaque
                );
                if ui.button("Apply to all channels").clicked() {
                    for color in &mut self.channel_colors {
                        *color = self.global_color;
                    }
                }
                ui.separator();

                if !self.eeg_info.ch_names.is_empty() {
                    egui::ComboBox::from_label("Select channel")
                        .selected_text(self.eeg_info.ch_names[self.selected_channel_for_color].clone())
                        .show_ui(ui, |ui| {
                            for (ch, name) in self.eeg_info.ch_names.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_channel_for_color, ch, name.clone());
                            }
                        });

                    if self.selected_channel_for_color < self.channel_colors.len() {
                        ui.label(format!("Color for {}:", self.eeg_info.ch_names[self.selected_channel_for_color]));
                        egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut self.channel_colors[self.selected_channel_for_color],
                            egui::color_picker::Alpha::Opaque
                        );
                    }
                }
                ui.separator();
                ui.heading("Measurement Ruler");
                ui.add(egui::Slider::new(&mut self.ruler_width, 0.1..=10.0).text("Width (s)"));
                ui.add(egui::Slider::new(&mut self.ruler_height, 10.0..=200.0).text("Height (µV)"));
                if ui.button("Place ruler").clicked() {

                    self.ruler_position = Some((self.x_view + 5.0, 0.0));
                }



            });


    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
