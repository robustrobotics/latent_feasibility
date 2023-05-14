from learning.evaluate.plot_compare_grasping_runs import plot_comparison_between_average_precision_of_two_experiments
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import compile_dataframes_and_save_path


def main():
    amortized_shapenet_exp_name = 'lamp-balanced'
    particle_shapenet_exp_name = 'lamp-balanced'

    amortized_shapenet_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_shapenet_exp_name, True)
    particle_shapenet_dataframe, _, _, _ = compile_dataframes_and_save_path(particle_shapenet_exp_name, True)
    plot_comparison_between_average_precision_of_two_experiments(amortized_shapenet_dataframe,
                                                                 particle_shapenet_dataframe,
                                                                 'amortized',
                                                                 'particle',
                                                                 'figures/')


if __name__ == '__main__':
    main()
