use super::*;

#[tokio::test]
async fn archived_results_keep_metadata_and_source_alpha_and_publish_only_final_reference_files()
-> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let original =
        image::RgbaImage::from_fn(8, 8, |x, y| image::Rgba([80, 90, 100, (x + y * 8) as u8]));
    let input = root.join("original.png");
    std::fs::write(&input, images::encode_png(&original, None)?)?;
    let studio = Studio::new(root.to_owned(), root.join("missing-python"))?;
    let metadata = json!({"seed":42,"lightning":"inference/models/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors","params":{"steps":32}});
    let job = {
        let mut store = studio.store.lock();
        let clean = image::RgbaImage::from_pixel(4, 4, image::Rgba([120, 130, 140, 255]));
        let clean =
            store
                .assets
                .save(&clean, "clean", "clean.png", false, Some(metadata.clone()))?;
        let result = store.assets.save(
            &original,
            "result",
            "result.png",
            false,
            Some(metadata.clone()),
        )?;
        json!({"seed":42,"transformer":"inference/models/Qwen-Image-2512-Q4_K_M.gguf","lightning":metadata["lightning"],"runtime":{"device":"cuda:0","precision":"bfloat16","offload":"sequential"},"clean":clean,"result":result})
    };
    let archive_directory = root.join("out/runs/test-run");
    std::fs::create_dir_all(&archive_directory)?;
    let archive = RunArchive {
        input: &input,
        directory: &archive_directory,
        stem: "original",
        run_id: "test-run",
    };
    let params = json!({"steps":32,"denoise":0.25,"passes":1,"restore_sigma":1.95,"restore":true,"restore_mode":"edge","unsharp":0.0,"keep_intermediate":true});
    let record = archive_result(&studio, &archive, 1, &params, &job)?;
    assert_eq!(record["transformer"], job["transformer"]);
    assert_eq!(record["lightning"], job["lightning"]);
    assert_eq!(record["runtime"]["device"], "cuda:0");
    assert!(
        !studio
            .store
            .lock()
            .assets
            .entries
            .contains_key(job["result"]["id"].as_str().unwrap())
    );
    assert!(
        !studio
            .store
            .lock()
            .assets
            .entries
            .contains_key(job["clean"]["id"].as_str().unwrap())
    );
    let result_path = Path::new(record["path"].as_str().context("Archived result missing")?);
    assert_eq!(
        images::png_metadata(&std::fs::read(result_path)?),
        Some(metadata)
    );
    let clean = images::open(&std::fs::read(
        record["clean_path"]
            .as_str()
            .context("Archived clean missing")?,
    )?)?;
    assert_eq!(clean.dimensions(), original.dimensions());
    assert!(
        clean
            .pixels()
            .zip(original.pixels())
            .all(|(clean, source)| clean.0[0..3] == [120, 130, 140] && clean[3] == source[3])
    );
    assert!(
        root.join("out/Original_desynth_s32_d0.250_p1_r1.95_edge.png")
            .is_file()
    );
    assert!(
        root.join("out/comparison_Original_desynth_s32_d0.250_p1_r1.95_edge.png")
            .is_file()
    );
    assert_eq!(
        std::fs::read_dir(root.join("out"))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .count(),
        2
    );
    studio.close().await;
    Ok(())
}

#[tokio::test]
async fn reference_aliases_cannot_overwrite_original_and_long_model_names_are_bounded() -> Result<()>
{
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let original = image::RgbaImage::from_pixel(8, 8, image::Rgba([80, 90, 100, 17]));
    let input = root.join("original.png");
    let original_bytes = images::encode_png(&original, None)?;
    std::fs::write(&input, &original_bytes)?;
    let studio = Studio::new(root.to_owned(), root.join("missing-python"))?;
    let job = {
        let mut store = studio.store.lock();
        let clean = store
            .assets
            .save(&original, "clean", "clean.png", false, None)?;
        let result = store
            .assets
            .save(&original, "result", "result.png", false, None)?;
        json!({"seed":42,"transformer":format!("custom-{}.gguf", "x".repeat(230)),"clean":clean,"result":result})
    };
    let archive_directory = root.join("out/runs/long-name");
    std::fs::create_dir_all(&archive_directory)?;
    let archive = RunArchive {
        input: &input,
        directory: &archive_directory,
        stem: "original",
        run_id: "long-name",
    };
    let params = json!({"steps":32,"denoise":0.25,"passes":1,"restore_sigma":1.95,"restore":true,"restore_mode":"edge","unsharp":0.0,"keep_intermediate":false});
    let alias = root.join("out/Original_desynth_s32_d0.250_p1_r1.95_edge.png");
    std::fs::hard_link(&input, &alias)?;
    let error = archive_result(&studio, &archive, 1, &params, &job).unwrap_err();
    assert!(error.to_string().contains("must not overwrite"));
    assert_eq!(std::fs::read(&input)?, original_bytes);
    assert_eq!(std::fs::read_dir(&archive_directory)?.count(), 0);
    std::fs::remove_file(alias)?;
    let record = archive_result(&studio, &archive, 1, &params, &job)?;
    let path = Path::new(record["path"].as_str().context("Archived result missing")?);
    assert!(path.is_file());
    assert!(path.file_name().unwrap().to_string_lossy().len() < 200);
    assert_eq!(std::fs::read(input)?, original_bytes);
    studio.close().await;
    Ok(())
}
