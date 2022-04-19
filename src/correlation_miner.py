from src.tigramite.pcmci import PCMCI
from src.tigramite.causal_effects import CausalEffects
from src.tigramite.independence_tests.cmisymb import CMIsymb
from src.tigramite import plotting as tp
import numpy as np
from matplotlib import pyplot as plt

class CausalDiscovery():

    def __init__(self, dataframe,
                 tau_max,
                 cond_ind_test,
                 pc_alpha = 0,
                 alpha_level = 0,
                 verbosity=0):
        self.dataframe = dataframe
        self.tau_max = tau_max
        self.cond_ind_test = cond_ind_test
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.cond_ind_test.set_dataframe(self.dataframe)
        self.verbosity = verbosity
        self.var_names = self.dataframe.var_names
        self.T = self.dataframe.T
        self.N = self.dataframe.N
    
    def initiate_stablePC(self):
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.cond_ind_test
        )
        all_parents, results = pcmci.run_pc_stable(pc_alpha=self.pc_alpha, tau_max=self.tau_max, max_combinations=5)
        if self.verbosity > 0:
            print("# of records: {}".format(pcmci.T))
            pcmci.print_significant_links(
                graph = results['graph'],
                p_matrix = results['p_matrix'], 
                val_matrix = results['val_matrix'],
                conf_matrix = results['conf_matrix'],
                alpha_level = self.pc_alpha
            )
        # NOTE: Followings are testing codes
        return all_parents, results

    def initiate_PCMCI(self):
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.cond_ind_test
        )
        results = pcmci.run_pcmci(tau_min=1, tau_max=self.tau_max, pc_alpha=self.pc_alpha, alpha_level=self.alpha_level)
        if self.verbosity > 0:
            print("# of records: {}".format(pcmci.T))
            pcmci.print_significant_links(
                graph = results['graph'],
                p_matrix = results['p_matrix'], 
                val_matrix = results['val_matrix'],
                conf_matrix = results['conf_matrix'],
                alpha_level = self.alpha_level
            )
        # NOTE: Followings are testing codes
        return results

class CausalInference():

    def __init__(self, dataframe, time_series_graph, X, Y, S):
        self.dataframe = dataframe
        self.time_series_graph = time_series_graph
        self.X = X
        self.Y = Y
        self.S = S
        self.causal_effects = CausalEffects(time_series_graph, graph_type='stationary_dag', X=self.X, Y=self.Y, S=self.S,
                           hidden_variables=None,
                           verbosity=0)
        
        self.opt = None
    
    def check__XYS_paths(self):
        newX, newY = self.causal_effects.check_XYS_paths()
        # self.X = newX
        # self.Y = newY

    def get_optimal_set(self):
        opt = self.causal_effects.get_optimal_set()
        # print("Oset = ", [(self.dataframe.var_names[v[0]], v[1]) for v in opt])
        self.opt = opt
    
    def check_optimality(self):
        optimality = self.causal_effects.check_optimality()
        return optimality

    def intervention_effect_prediction(self, intervention_data):
        """
        Calculate the effect on Y given the intervention of X (with conditioning set self.S and the optimal adjustment set self.opt)
        For instance of Z + instance of X (according to intervention_data), calculate the expectation value of Y.

        Note that we only consider the case of single outcome. Specifically, here we assume len(Y) == 1.

        Args:
            intervention_data (list[int]): The intervention value for all variables in X
        """
        self.check__XYS_paths()
        self.get_optimal_set()
        assert(self.check_optimality())
        assert(len(self.X) == len(intervention_data))
        assert(len(self.Y) == 1)

        array, xyz =  \
            self.dataframe.construct_array(X=self.X, Y=self.Y, # + self.Z, 
                                           Z=self.S,
                                           extraZ=self.opt,
                                           tau_max=self.causal_effects.tau_max,
                                           mask_type=None,
                                           cut_off='tau_max',
                                           verbosity=1)
        array = array.T # transpose from N * T matrix to T * N matrix
        X_indices = list(np.where(xyz==0)[0])
        for i in range(len(X_indices)): # For each intervention variable indexed by X_indices[i], its value is set to be intervention_data[i]
            array = array[ array[:, X_indices[i]] == intervention_data[i] ] # Select those columns which cause variables are fixed to the intervention value
        array = array.T # transpose back to N * T matrix
        predictor_indices =  list(np.where(xyz==3)[0]) \
                           + list(np.where(xyz==2)[0])
        predictor_array = array[predictor_indices, :].T
        # Given the intervention_data, further process the predictor_array
        target_array = array[np.where(xyz==1)[0][0], :].T
        fre_dict = {}
        row_count = 0
        for row in predictor_array:
            key = ''.join([str(x) for x in list(row)]) # The key is instance of Z + extraZ
            fre_dict[key] = [0, 0] if key not in fre_dict.keys() else fre_dict[key]
            fre_dict[key][0] += 1; fre_dict[key][1] += target_array[row_count] # Update the [instance #, summed y] pair
            row_count += 1
        frequency_table = list(fre_dict.values())
        weighted_frequency_table = [(entry[0] * 1.0 / row_count, entry[1] * 1.0 / entry[0]) for entry in frequency_table]
        return sum([val[0] * val[1] for val in weighted_frequency_table])

class CorrelationMiner():

    def __init__(self, dataframe, discovery_method='stable-pc'):
        self.dataframe = dataframe
        self.discovery_method = discovery_method 
        self.all_parents = None
        self.discovery_results = None

    def initiate_causal_discovery(self, tau_max=1, pc_alpha=0, alpha_level=0):
        all_parents = None; results = None
        causal_miner = CausalDiscovery(dataframe=self.dataframe, tau_max=tau_max, cond_ind_test=CMIsymb(
            significance='shuffle_test', n_symbs= None
            ), pc_alpha=pc_alpha, alpha_level=alpha_level,
             verbosity=1
            )
        if self.discovery_method == 'stable-pc':
            all_parents, results = causal_miner.initiate_stablePC()
            self.all_parents = all_parents
            self.discovery_results = results
        elif self.discovery_method == 'pcmci':
            self.discovery_results = causal_miner.initiate_PCMCI()

    def initiate_causal_inference(self, tau_max):
        assert(self.discovery_results is not None)
        time_series_graph = self.discovery_results['graph'] # First derive the DAG time_series graph
        # if self.discovery_method == 'stable-pc':
        #     assert((self.all_parents is not None) and (self.discovery_results is not None))
        #     time_series_graph = CausalEffects.get_graph_from_dict(self.all_parents, tau_max = tau_max)
        # assert(time_series_graph is not None)
        effects_dict = {} # effects_dict[cause_attr][outcome_attr][intervention] stores the corresponding effect.
        for key in self.all_parents.keys(): # Traverse each causal-outcome pair in the graph and estimate the causal effects.
            outcome_attr = (key, 0)
            cause_attrs = self.all_parents[key]
            for cause_attr in cause_attrs:
                causal_inferencer = CausalInference(dataframe=self.dataframe, \
                    time_series_graph=time_series_graph, X=[cause_attr], Y=[outcome_attr], S=[]) # We did not test conditional ATEs; Moreover, we assume that X and Y only contains single attribute
                causal_inferencer.check__XYS_paths()
                assert(causal_inferencer.check_optimality()) # For DAGs, the optimality always holds.
                causal_inferencer.get_optimal_set()
                effects_dict[cause_attr] = {} if cause_attr not in effects_dict.keys() else effects_dict[cause_attr]
                effects_dict[cause_attr][outcome_attr] = {} if outcome_attr not in effects_dict[cause_attr].keys() else effects_dict[cause_attr][outcome_attr]
                effects_dict[cause_attr][outcome_attr][1] = causal_inferencer.intervention_effect_prediction([1])
                effects_dict[cause_attr][outcome_attr][0] = causal_inferencer.intervention_effect_prediction([0])
        return effects_dict
    
    def policy_specification(self, effects_dict):
        count = 0
        for cause_attr in effects_dict.keys():
            for outcome_attr in effects_dict[cause_attr].keys():
                if (effects_dict[cause_attr][outcome_attr][1] > 0.5 and effects_dict[cause_attr][outcome_attr][1] > 0.5) or (effects_dict[cause_attr][outcome_attr][1] <= 0.5 and effects_dict[cause_attr][outcome_attr][1] <= 0.5):
                    count += 1
        print(count)

if __name__ == '__main__':
    effects_dict = {(0, -1): {(0, 0): {1: 0.9391891891891893, 0: 0.0015126050420168065}, (27, 0): {1: 0.3918918918918919, 0: 0.06369747899159664}}, (1, -1): {(0, 0): {1: 0.1040056219255095, 0: 0.0}, (1, 0): {1: 0.9803232607167954, 0: 0.005775401069518717}, (2, 0): {1: 0.025298664792691498, 0: 0.32641711229946535}, (10, 0): {1: 0.09065354884047787, 0: 0.45518716577540114}, (12, 0): {1: 0.2986647926914968, 0: 0.43101604278074873}, (14, 0): {1: 0.1033028812368236, 0: 0.3835294117647059}, (15, 0): {1: 0.43851018973998596, 0: 0.5392513368983958}, (18, 0): {1: 0.24033731553056925, 0: 0.021176470588235293}, (20, 0): {1: 0.037947997189037234, 0: 0.06310160427807485}, (22, 0): {1: 0.11314125087842586, 0: 0.02053475935828877}, (25, 0): {1: 0.019676739283204497, 0: 0.2536898395721926}, (27, 0): {1: 0.2227687983134223, 0: 0.025668449197860963}, (28, 0): {1: 0.06535488404778636, 0: 0.016470588235294115}, (29, 0): {1: 0.004216444132115249, 0: 0.05219251336898395}, (30, 0): {1: 0.060435699226985246, 0: 0.6344385026737968}}, (15, -1): {(0, 0): {1: 0.0003180661577608142, 0: 0.049763033175355444}, (10, 0): {1: 0.7143765903307887, 0: 0.003723764387271496}, (12, 0): {1: 0.7713104325699746, 0: 0.0050778605280974946}, (15, 0): {1: 0.9942748091603054, 0: 0.006431956668923495}, (29, 0): {1: 0.00826972010178117, 0: 0.07582938388625592}, (30, 0): {1: 0.6278625954198473, 0: 0.36492890995260663}}, (16, -1): {(0, 0): {1: 0.01911544227886057, 0: 0.0282798833819242}, (7, 0): {1: 0.04310344827586207, 0: 0.01282798833819242}, (11, 0): {1: 0.2668665667166416, 0: 0.2285714285714286}, (15, 0): {1: 0.535232383808096, 0: 0.5005830903790087}, (16, 0): {1: 0.9962518740629686, 0: 0.003206997084548105}, (18, 0): {1: 0.0648425787106447, 0: 0.07813411078717201}, (22, 0): {1: 0.06634182908545726, 0: 0.023323615160349864}}, (31, -1): {(1, 0): {1: 0.9391304347826086, 0: 0.21962226307872304}, (5, 0): {1: 0.026086956521739126, 0: 0.2121009526993147}, (14, 0): {1: 0.2, 0: 0.3204078221627945}, (31, 0): {1: 0.991304347826087, 0: 0.0}, (32, 0): {1: 0.4782608695652174, 0: 0.18686277787063346}}, (13, -1): {(1, 0): {1: 0.04301075268817204, 0: 0.3292201382033564}, (10, 0): {1: 0.9916911045943303, 0: 0.05626850937808489}, (12, 0): {1: 0.9486803519061583, 0: 0.12314906219151035}, (13, 0): {1: 0.9960899315738025, 0: 0.002221125370187561}, (19, 0): {1: 0.04545454545454544, 0: 0.029368213228035536}}, (4, -1): {(1, 0): {1: 0.38604353393085783, 0: 0.0726294552790854}, (3, 0): {1: 0.9324583866837388, 0: 0.04236718224613316}, (4, 0): {1: 0.9964788732394366, 0: 0.004034969737726967}, (17, 0): {1: 0.9615877080665813, 0: 0.18594485541358438}, (28, 0): {1: 0.034250960307298337, 0: 0.02118359112306658}}, (18, -1): {(1, 0): {1: 0.7346938775510203, 0: 0.1940958105002652}, (2, 0): {1: 0.1337868480725624, 0: 0.26568852748806787}, (18, 0): {1: 0.5170068027210885, 0: 0.03765246597136289}, (22, 0): {1: 0.0022675736961451248, 0: 0.04525366802191972}, (25, 0): {1: 0.0045351473922902496, 0: 0.21424783454127636}, (26, 0): {1: 0.006802721088435375, 0: 0.023157150433091744}, (27, 0): {1: 0.3401360544217687, 0: 0.050733604383949096}}, (2, -1): {(2, 0): {1: 0.9807938540332907, 0: 0.006613756613756613}, (11, 0): {1: 0.8905249679897566, 0: 0.023148148148148147}, (18, 0): {1: 0.033930857874519854, 0: 0.0855379188712522}}, (14, -1): {(2, 0): {1: 0.7670103092783505, 0: 0.017797017797017794}, (6, 0): {1: 0.8262886597938144, 0: 0.007936507936507936}, (11, 0): {1: 0.7628865979381442, 0: 0.003848003848003848}, (13, 0): {1: 0.9154639175257732, 0: 0.06517556517556518}, (14, 0): {1: 0.9835051546391753, 0: 0.007696007696007697}}, (32, -1): {(2, 0): {1: 0.6254266211604095, 0: 0.1682907023954527}, (19, 0): {1: 0.011092150170648464, 0: 0.04039788875355259}, (20, 0): {1: 0.08959044368600681, 0: 0.049533089727974024}, (21, 0): {1: 0.0068259385665529, 0: 0.030247665448639872}, (25, 0): {1: 0.2960750853242321, 0: 0.17600487210718627}, (32, 0): {1: 0.9974402730375427, 0: 0.0008120178643930167}}, (3, -1): {(3, 0): {1: 0.9983541803818302, 0: 0.00196078431372549}, (4, 0): {1: 0.9598420013166558, 0: 0.06830065359477124}}, (17, -1): {(3, 0): {1: 0.8214285714285714, 0: 0.046420141620771044}, (4, 0): {1: 0.8456130483689539, 0: 0.046420141620771044}, (5, 0): {1: 0.32367829021372324, 0: 0.04760031471282455}, (17, 0): {1: 0.9952193475815523, 0: 0.007081038552321006}, (20, 0): {1: 0.08605174353205851, 0: 0.016915814319433516}, (29, 0): {1: 0.011248593925759281, 0: 0.08261211644374511}}, (6, -1): {(4, 0): {1: 0.6363080684596577, 0: 0.4670551322277006}, (5, 0): {1: 0.7665036674816627, 0: 0.004034065441506051}, (6, 0): {1: 0.9908312958435207, 0: 0.003361721201255043}}, (5, -1): {(5, 0): {1: 0.9897798742138366, 0: 0.0026937422295897225}}, (9, -1): {(6, 0): {1: 0.5355029585798816, 0: 0.0022905759162303663}, (9, 0): {1: 0.9947403024326101, 0: 0.00556282722513089}, (10, 0): {1: 0.692965154503616, 0: 0.048756544502617793}, (14, 0): {1: 0.6130834976988824, 0: 0.02454188481675393}, (22, 0): {1: 0.07363576594345828, 0: 0.010798429319371729}}, (10, -1): {(6, 0): {1: 0.7207446808510638, 0: 0.002602811035918792}, (10, 0): {1: 0.9929078014184397, 0: 0.004424778761061947}, (13, 0): {1: 0.900709219858156, 0: 0.0039042165538781884}, (15, 0): {1: 0.9977836879432624, 0: 0.23269130661114004}}, (7, -1): {(7, 0): {1: 0.8364779874213836, 0: 0.004377841387438962}, (20, 0): {1: 0.610062893081761, 0: 0.04243138575517765}, (23, 0): {1: 0.6415094339622641, 0: 0.008924061289779423}, (25, 0): {1: 0.0, 0: 0.2044115170904193}}, (23, -1): {(7, 0): {1: 0.6064516129032258, 0: 0.010937237085646978}, (20, 0): {1: 0.5225806451612903, 0: 0.045095069830052174}, (23, 0): {1: 0.6774193548387097, 0: 0.008413259296651522}, (25, 0): {1: 0.0, 0: 0.2042739357226989}}, (8, -1): {(8, 0): {1: 0.9764150943396226, 0: 0.0008494733265375468}, (24, 0): {1: 0.2028301886792453, 0: 0.0018688413183826026}}, (24, -1): {(8, 0): {1: 0.7962962962962963, 0: 0.027961614824619455}, (24, 0): {1: 0.7222222222222222, 0: 0.002481800132362674}}, (11, -1): {(11, 0): {1: 0.9866310160427807, 0: 0.004345936549326379}, (14, 0): {1: 0.9906417112299465, 0: 0.09952194697957409}, (27, 0): {1: 0.03141711229946525, 0: 0.0847457627118644}}, (27, -1): {(11, 0): {1: 0.11670480549199083, 0: 0.2552552552552553}, (18, 0): {1: 0.35697940503432485, 0: 0.05034446210916801}, (19, 0): {1: 0.009153318077803204, 0: 0.0367426249779191}, (21, 0): {1: 0.002288329519450801, 0: 0.02755696873343932}, (25, 0): {1: 0.0, 0: 0.21444974386150853}, (26, 0): {1: 0.004576659038901602, 0: 0.023317435082140965}, (27, 0): {1: 0.5446224256292908, 0: 0.035152799858682214}, (28, 0): {1: 0.011441647597254004, 0: 0.029146793852676205}, (29, 0): {1: 0.0, 0: 0.04416180886769121}}, (12, -1): {(12, 0): {1: 0.993029930299303, 0: 0.004919376878928669}, (22, 0): {1: 0.06601066010660105, 0: 0.02623667668761956}}, (33, -1): {(16, 0): {1: 0.025601241272304114, 0: 0.5481389062175089}, (33, 0): {1: 0.9875872769588827, 0: 0.0033270950301517986}, (34, 0): {1: 0.9705197827773467, 0: 0.022249948014140153}}, (28, -1): {(16, 0): {1: 0.5411764705882353, 0: 0.4347165991902834}, (22, 0): {1: 0.0, 0: 0.043353576248313104}, (25, 0): {1: 0.0, 0: 0.20479082321187578}, (28, 0): {1: 0.34705882352941175, 0: 0.018724696356275307}, (29, 0): {1: 0.0, 0: 0.04217273954116062}}, (30, -1): {(18, 0): {1: 0.03997378768020969, 0: 0.10472751149047933}, (30, 0): {1: 0.9950851900393185, 0: 0.004924491135915956}}, (19, -1): {(19, 0): {1: 0.490566037735849, 0: 0.018348623853211007}, (20, 0): {1: 0.23113207547169812, 0: 0.050968399592252814}, (21, 0): {1: 0.25943396226415094, 0: 0.017329255861365956}, (25, 0): {1: 0.009433962264150945, 0: 0.20591233435270126}}, (21, -1): {(19, 0): {1: 0.3375796178343949, 0: 0.02676317118330247}, (21, 0): {1: 0.36942675159235666, 0: 0.016663861302810976}, (22, 0): {1: 0.0, 0: 0.04325871065477191}, (25, 0): {1: 0.0, 0: 0.2043427032486113}}, (25, -1): {(19, 0): {1: 0.0, 0: 0.04340704340704342}, (20, 0): {1: 0.0016474464579901153, 0: 0.07104832104832105}, (21, 0): {1: 0.0, 0: 0.032145782145782155}, (22, 0): {1: 0.0065897858319604605, 0: 0.050982800982800995}, (25, 0): {1: 0.018945634266886325, 0: 0.24385749385749383}, (26, 0): {1: 0.0, 0: 0.02743652743652744}, (27, 0): {1: 0.0016474464579901153, 0: 0.08906633906633908}, (28, 0): {1: 0.0, 0: 0.034807534807534804}, (29, 0): {1: 0.0, 0: 0.051187551187551215}, (30, 0): {1: 0.6400329489291599, 0: 0.46580671580671573}}, (20, -1): {(20, 0): {1: 0.3954154727793696, 0: 0.0367020351365455}, (23, 0): {1: 0.22636103151862463, 0: 0.013219690380935815}, (25, 0): {1: 0.005730659025787966, 0: 0.21081927291702907}, (26, 0): {1: 0.0, 0: 0.023308401461123673}}, (22, -1): {(22, 0): {1: 0.2645914396887159, 0: 0.03235747303543913}, (25, 0): {1: 0.038910505836575876, 0: 0.20612908748501965}}, (29, -1): {(25, 0): {1: 0.0, 0: 0.20759233926128598}, (29, 0): {1: 0.068, 0: 0.03984268125854992}}, (26, -1): {(25, 0): {1: 0.0, 0: 0.20355466130114022}, (26, 0): {1: 0.3432835820895523, 0: 0.014755197853789401}, (29, 0): {1: 0.14925373134328357, 0: 0.03856472166331323}}, (34, -1): {(29, 0): {1: 0.012518409425625921, 0: 0.0491561181434599}, (33, 0): {1: 0.9212076583210603, 0: 0.008016877637130802}, (34, 0): {1: 0.9948453608247422, 0: 0.0014767932489451478}}}
    correlation_miner = CorrelationMiner(dataframe=None)
    correlation_miner.policy_specification(effects_dict=effects_dict)