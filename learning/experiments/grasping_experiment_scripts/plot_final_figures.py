from learning.evaluate.plot_compare_grasping_runs import plot_comparison_between_average_precision_of_two_experiments
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import compile_dataframes_and_save_path


def main():
    # amortized_exp_name = 'gnp-snv2scaleddim-run10'
    # particle_exp_name = 'pf-snv2scaleddim-run10'
    # amortized_name = 'sn-amortized'
    # particle_name = 'sn-particle'
    amortized_exp_name = 'gnp-boxv2-run1'
    particle_exp_name = 'pf-boxv2-run1'
    amortized_name = 'box-amortized'
    particle_name = 'box-particle'

    amortized_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_exp_name, True)
    particle_dataframe, _, _, _ = compile_dataframes_and_save_path(particle_exp_name, True)
    plot_comparison_between_average_precision_of_two_experiments(
        amortized_dataframe,
        particle_dataframe,
        amortized_name,
        particle_name,
        'figures/',
        'regret'
    )
    plot_comparison_between_average_precision_of_two_experiments(
        amortized_dataframe,
        particle_dataframe,
        amortized_name,
        particle_name,
        'figures/',
        'average precision'
    )


if __name__ == '__main__':
    main()
