from learning.evaluate.plot_compare_grasping_runs import plot_comparison_between_average_precision_of_two_experiments, \
    plot_comparison_between_average_precision_of_n_experiments
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import compile_dataframes_and_save_path


def main():
    amortized_exp_name = 'gnp-snv2scaleddim-ensemble-run1'
    particle_exp_name = 'pf10000-snv2scaleddim-ensemble-run1'
    # particle_exp_name = 'gnp-snv2scaleddim-ensemble-run1'

    amortized_name = 'sn-amortized'
    particle_name = 'sn-particle'
    # amortized_exp_name = 'gnp-boxv2-run1'
    # particle_exp_name = 'pf-boxv2-run1'
    # amortized_name = 'box-amortized'
    # particle_name = 'box-particle'

    # ---- Main Experiments ----
    amortized_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_exp_name, True)
    particle_dataframe, _, _, _ = compile_dataframes_and_save_path(particle_exp_name, True)
    # plot_comparison_between_average_precision_of_two_experiments(
    #     amortized_dataframe,
    #     particle_dataframe,
    #     amortized_name,
    #     particle_name,
    #     'figures/',
    #     'regret'
    # )
    plot_comparison_between_average_precision_of_two_experiments(
        amortized_dataframe,
        particle_dataframe,
        amortized_name,
        particle_name,
        'figures/',
        'average precision'
    )

    # particle performance shapenet
    sn_10_exp = 'pf10-snv2scaledim-run10'
    sn_100_exp = 'pf100-snv2scaledim-run10'
    sn_1000_exp = 'pf-snv2scaleddim-run10'
    sn_10000_exp = 'pf10000-snv2scaleddim-run10'
    sn_list = [sn_100_exp, sn_1000_exp, sn_10000_exp]
    sn_dfs = [compile_dataframes_and_save_path(sn_exp, True)[0] for sn_exp in sn_list]
    sn_names = ['sn_100_parts', 'sn_1000_parts', 'sn_10000_parts']
    plot_comparison_between_average_precision_of_n_experiments(sn_dfs, sn_names, 'figures/', 'average precision')

    # boxes performance shapenet
    bx_10_exp = 'pf10-boxv2-run1'
    bx_100_exp = 'pf100-boxv2-run1'
    bx_1000_exp = 'pf-boxv2-run1'
    bx_list = [bx_10_exp, bx_100_exp, bx_1000_exp]
    bx_dfs = [compile_dataframes_and_save_path(bx_exp, True)[0] for bx_exp in bx_list]
    bx_names = ['bx_10_parts', 'bx_100_parts', 'bx_1000_parts']
    plot_comparison_between_average_precision_of_n_experiments(bx_dfs, bx_names, 'figures/', 'average precision')



if __name__ == '__main__':
    main()
