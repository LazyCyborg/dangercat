

pub fn compute_average_reference_f32(data: &Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let num_channels = data.len();
    let num_time_points = data[0].len();
    let mut average_ref = vec![vec![0.0; num_time_points]; num_channels];

    for t in 0..num_time_points {
        let mut average = 0.0;
        for ch in 0..num_channels {
            average += data[ch][t];
        }
        average /= num_channels as f32;

        for ch in 0..num_channels {
            average_ref[ch][t] = data[ch][t] - average;
        }
    }

    Ok(average_ref)
}

pub fn compute_average_reference_i16(data: &Vec<Vec<i16>>) -> Result<Vec<Vec<i16>>, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let num_channels = data.len();
    let num_time_points = data[0].len();
    let mut average_ref = vec![vec![0; num_time_points]; num_channels];

    for t in 0..num_time_points {
        let mut average = 0.0;
        for ch in 0..num_channels {
            average += data[ch][t] as f64;
        }
        average /= num_channels as f64;

        for ch in 0..num_channels {
            average_ref[ch][t] = (data[ch][t] as f64 - average).round() as i16;
        }
    }

    Ok(average_ref)
}
