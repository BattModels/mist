from ternary_plots import plot_vector_fields
import glob


if __name__ == "__main__":
    run_id = "8aqjzwmr"  # "z4ni8hcj" # "c7ssptg7"
    pretrained_ckpt = (
        f"/home/abhutani/electrolyte-fm/mist/{run_id}/checkpoints/last.ckpt"
    )
    # for target in ["transferance", "conductivity"]:
    #     for salt_composition in [0.05, 0.1, 0.15]:
    #         for salt in ["LiPF6", "TFSI"]:
    #             dir_name = f"{salt}_{int(salt_composition*100)}"
    #             generate_dataset(
    #                 salt_mole_fraction=salt_composition, save_dir=dir_name, salt=salt
    #             )
    #             run_inference(
    #                 pretrained_ckpt,
    #                 data_dir=dir_name,
    #                 val_batch_size=165,
    #                 target=target,
    #             )
    for target in [
        # "transferance",
        "conductivity",
        # "product",
    ]:  # "diffusioncoeff"]:
        plot_vector_fields(data_files=glob.glob(f"{target}_*.csv"), target=target)
