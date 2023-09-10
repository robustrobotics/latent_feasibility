from learning.evaluate.plot_compare_grasping_runs import plot_comparison_between_two_experiments, \
    plot_comparison_between_n_experiments
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import compile_dataframes_and_save_path
import sys
import pickle

def main():
    amortized_exp_name = 'gnp-boxv2-run1'
    amortized_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_exp_name, True)
    with open(f'df_{amortized_exp_name}.pkl', 'wb') as handle:
        pickle.dump(amortized_dataframe, handle)

    import ipdb; ipdb.set_trace()
    amortized_exp_name = 'gnp-snv2scaleddim-run10'
    amortized_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_exp_name, True)
    with open(f'df_{amortized_exp_name}.pkl', 'wb') as handle:
        pickle.dump(amortized_dataframe, handle)

    sys.exit()
    # known_particle_sn_name = 'pf10000-snv2scaleddim-run10'
    # known_particle_sn_name = 'pf10000-boxv2-updated-dist'
    # known_particle_bx_name = 'pf10000-boxv2-updated-dist'
    known_particle_sn_name = 'gnp-snv2scaleddim-run10'
    known_particle_bx_name = 'gnp-snv2scaleddim-run10'

    sn_known_df, _, _, _ = compile_dataframes_and_save_path(known_particle_sn_name, True)
    bx_known_df, _, _, _ = compile_dataframes_and_save_path(known_particle_bx_name, True)

    plot_comparison_between_two_experiments(sn_known_df, bx_known_df, 'gnp',
                                            'learned pf 10k',
                                            'figures/', 'average precision')

    # plot_comparison_between_two_experiments(sn_known_df, sn_known_df, known_particle_sn_name,
    #                                         known_particle_sn_name + '_dup',
    #                                         'figures/', 'precision', metric_name_opt='recall')
    # plot_comparison_between_two_experiments(bx_known_df, bx_known_df, known_particle_bx_name,
    #                                         known_particle_bx_name + '_dup',
    #                                         'figures/', 'precision', metric_name_opt='recall')

    sys.exit()
    amortized_exp_name = 'gnp-snv2scaleddim-run10'
    particle_exp_name = 'pf10000-snv2scaleddim-run10'
    amortized_name = 'sn-amortized'
    particle_name = 'sn-particle'
    # amortized_exp_name = 'gnp-boxv2-run1'
    # particle_exp_name = 'pf-boxv2-run1'
    # amortized_name = 'box-amortized'
    # particle_name = 'box-particle'

    # ---- Main Experiments ----
    amortized_dataframe, _, _, _ = compile_dataframes_and_save_path(amortized_exp_name, True)
    particle_dataframe, _, _, _ = compile_dataframes_and_save_path(particle_exp_name, True)
    # plot_comparison_between_two_experiments(
    #     amortized_dataframe,
    #     particle_dataframe,
    #     amortized_name,
    #     particle_name,
    #     'figures/',
    #     'regret'
    # )
    plot_comparison_between_two_experiments(
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
    plot_comparison_between_n_experiments(sn_dfs, sn_names, 'figures/', 'average precision')

    # boxes performance shapenet
    bx_10_exp = 'pf10-boxv2-run1'
    bx_100_exp = 'pf100-boxv2-run1'
    bx_1000_exp = 'pf-boxv2-run1'
    bx_list = [bx_10_exp, bx_100_exp, bx_1000_exp]
    bx_dfs = [compile_dataframes_and_save_path(bx_exp, True)[0] for bx_exp in bx_list]
    bx_names = ['bx_10_parts', 'bx_100_parts', 'bx_1000_parts']
    plot_comparison_between_n_experiments(bx_dfs, bx_names, 'figures/', 'average precision')


if __name__ == '__main__':
    main()
