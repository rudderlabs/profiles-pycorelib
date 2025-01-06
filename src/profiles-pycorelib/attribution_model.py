from typing import List, Dict, Tuple, Union
from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.schema import ContractBuildSpecSchema, EntityKeyBuildSpecSchema, EntityIdsBuildSpecSchema, MaterializationBuildSpecSchema
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger
import seaborn as sns
import matplotlib
import scipy.linalg as linalg
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import pathlib

matplotlib.use('agg')
MAXIMUM_TOUCHPOINTS_TO_VISUALIZE = 50

class AttributionModel(BaseModelType):
    TypeName = "campaign_attribution_scores" # the name of the model type

    # json schema for the build spec
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "touchpoint_var" : { "type": "string" }, # model
            "days_since_first_seen_var": { "type": "string" }, # model
            "first_seen_since": { "type": "integer" },
            "conversion_entity_var": { "type": "string" }, #model 
            "enable_visualisation": { "type": "boolean" },
            "entity": { "type": "string"}
            # extend your schema from pre-defined schemas
            # **ContractBuildSpecSchema["properties"],
            # **EntityKeyBuildSpecSchema["properties"],
            # **EntityIdsBuildSpecSchema["properties"],
            # **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["entity", "touchpoint_var"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        # call the base class constructor
        # it will initialize contract, entity_key, ids and materialization, if present in the build_spec
        super().__init__(build_spec, schema_version, pb_version)

        # override the values from the build_spec
        # self.entity_key = "user"

    def get_material_recipe(self)-> PyNativeRecipe:
        # return the recipe object
        return AttributionModelRecipe(self.build_spec)
    
    def validate(self) -> Tuple[bool, str]:
        return True, "Validated successfully"

class MultiTouchModels:
    def __init__(self, logger):
        self.logger = logger

    def linear_model(self, input_df, touchpoints_array_col, conversion_col:str) -> pd.DataFrame:
        input_df = input_df.copy()
        input_df['n_touches'] = input_df[touchpoints_array_col].apply(lambda x: len(x) if x else 0)
        input_df['weight'] = input_df.apply(lambda row: row[conversion_col]/row['n_touches'] if row['n_touches']>0 else 0, axis=1)
        linear_scores = input_df.explode(touchpoints_array_col).groupby(touchpoints_array_col).agg({'weight':'sum'}).reset_index()
        linear_scores.columns = [touchpoints_array_col, "linear_score"]
        del input_df['n_touches'], input_df['weight']
        return linear_scores
    
    def _generate_transition_counts(self, journey_list: List[List[Union[int, str]]], 
                                distinct_touches_list: List[Union[int, str]], 
                                is_positive: bool,
                                journey_weights: List[float] = None):
        if is_positive:
            destination_idx = -1
        else:
            destination_idx = -2
        transition_counts = np.zeros(((len(distinct_touches_list)+3), (len(distinct_touches_list)+3)))
        for journey_id, journey in enumerate(journey_list):
            journey_weight = journey_weights[journey_id] if is_positive and journey_weights else 1
            transition_counts[0, (distinct_touches_list.index(journey[0])+1)] += journey_weight # First point in the path
            for n, touch_point in enumerate(journey):
                if n == len(journey) - 1:
                    # Reached last point
                    transition_counts[(distinct_touches_list.index(touch_point)+1), destination_idx] += journey_weight
                    transition_counts[destination_idx, destination_idx]+=journey_weight
                else:
                    transition_counts[(distinct_touches_list.index(touch_point)+1), (distinct_touches_list.index(journey[n+1]) + 1)] +=journey_weight
        transition_labels = distinct_touches_list.copy()
        transition_labels.insert(0, "Start")
        transition_labels.extend(["Dropoff", "Converted"])
        return transition_counts, transition_labels
    
    def _plot_transitions(self, 
                          transition_probabilities: np.array, 
                          labels: List[Union[int, str]], 
                          image_path, 
                          title="User Journey Map"):
        n_labels = len(labels)
        color_hex_codes = [mcolors.to_hex(plt.cm.viridis(i/n_labels)) for i in range(n_labels)]
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=color_hex_codes
            ),
            link=dict(
                source=[i for i in range(len(labels)) for j in range(len(labels)) if transition_probabilities[i][j] > 0 and i != j],
                target=[j for i in range(len(labels)) for j in range(len(labels)) if transition_probabilities[i][j] > 0 and i != j],
                value=[transition_probabilities[i][j] for i in range(len(labels)) for j in range(len(labels)) if transition_probabilities[i][j] > 0 and i != j]
            )
        )])

        fig.update_layout(title_text=f"<b>{title}</b>", font_size=10, title_x=0.5)
        fig.write_html(image_path)
    
    def _get_transition_probabilities(self, 
                                    converted_touchpoints_list: List[List[Union[int, str]]], 
                                    dropoff_touchpoints_list: List[List[Union[int, str]]], 
                                    distinct_touches_list: List[Union[int, str]],
                                    journey_weights: List[float] = None) -> Tuple[np.array, List[Union[int, str]]]:
        row_normalize_np_array = lambda transition_counts: np.where(transition_counts.sum(axis=1)[:, np.newaxis] != 0,
                                                            transition_counts / transition_counts.sum(axis=1)[:, np.newaxis],
                                                            0)
        pos_transitions, _ = self._generate_transition_counts(converted_touchpoints_list, distinct_touches_list, is_positive=True, journey_weights=journey_weights)
        neg_transitions, labels = self._generate_transition_counts(dropoff_touchpoints_list, distinct_touches_list, is_positive=False)
        all_transitions = pos_transitions + neg_transitions
        transition_probabilities = row_normalize_np_array(all_transitions)
        return transition_probabilities, labels

    def _converge(self, transition_matrix, max_iters=40, verbose=False):
        transition_matrix = np.array(transition_matrix, dtype=np.float32)
        T_upd = transition_matrix.copy()
        prev_T = transition_matrix.copy()
        for i in range(max_iters):
            T_upd = linalg.blas.dgemm(1.0, transition_matrix, prev_T)
            if np.abs(T_upd - prev_T).max() < 1e-3:
                if verbose:
                    self.logger.info(f"{i} iters taken for convergence")
                return T_upd
            prev_T = T_upd
        if verbose:
            self.logger.info(f"Max iters of {max_iters} reached before convergence. Exiting")
        return T_upd
    
    def _get_removal_affects(self, transition_probs, labels, ignore_labels=["Start", "Dropoff","Converted"], default_conversion=1.):
        removal_affect = {}
        for n, label in enumerate(labels):
            if n % max((len(labels)//10), 1) == 0:
                self.logger.info(f"Computing removal affects: {n/len(labels)*100:.2f}%")
            if label in ignore_labels:
                continue
            else:
                drop_transition = transition_probs.copy()
                drop_transition[n,:] = 0. # Drop all transitions from this touchpoint
                drop_transition[n,-2] = 1. # Force all touches to dropoff from this touchpoint
                drop_transition_converged = self._converge(drop_transition)
                removal_affect[label] = default_conversion - drop_transition_converged[0,-1]
        return removal_affect
    
    def _converged_transition_probabilities(self, transition_probabilities):
        tic = time.time()
        converged_transition_probabilities = self._converge(transition_probabilities)
        toc = time.time()
        time_per_iter = (toc - tic)
        total_iters = transition_probabilities.shape[0]
        eta =  time_per_iter * total_iters
        if eta > 300:
            self.logger.info(f"Computing Markov score. This step may take a while. ETA: {eta: .2f} seconds")
        return converged_transition_probabilities
    
    def _get_markov_scores(self, 
                            tp_list_positive: List[List[Union[int, str]]],
                            tp_list_negative: List[List[Union[int, str]]], 
                            distinct_touches_list: List[str],
                            enable_visualisation: bool,
                            journey_weights: List[float]=None,
                            attribution_reports_folder_path:str=None,) -> Tuple[Dict[Union[int, str], float], np.array]:
        transition_probabilities, labels = self._get_transition_probabilities(tp_list_positive, tp_list_negative, distinct_touches_list, journey_weights=journey_weights)
        transition_probabilities_converged = self._converged_transition_probabilities(transition_probabilities)
        if attribution_reports_folder_path and enable_visualisation:
            image_file = os.path.join(attribution_reports_folder_path, "user_journey_map.html")
            self._plot_transitions(transition_probabilities, labels, image_file)
        removal_affects = self._get_removal_affects(transition_probabilities, labels, default_conversion=transition_probabilities_converged[0,-1])
        total_conversions = sum(journey_weights) if journey_weights else len(tp_list_positive)
        attributable_conversions = {}
        total_weight = sum(removal_affects.values())
        for tp, weight in removal_affects.items():
            attributable_conversions[tp] = weight/total_weight * total_conversions
        return attributable_conversions
    
    def get_markov_attribution(self, input_df: pd.DataFrame, conversion_col: str, touchpoints_array_col: str, attribution_reports_folder_path: str, enable_visualisation: bool) -> pd.DataFrame:
        data_filtered = input_df[input_df[touchpoints_array_col].apply(lambda touches: len(touches)>0 if touches else False)]
        positive_touchpoints_ = data_filtered.query(f"{conversion_col}>0")[[touchpoints_array_col, conversion_col]].values
        positive_touches = [val[0] for val in positive_touchpoints_]
        conversion_weights = [val[1] for val in positive_touchpoints_]
        negative_touchpoints_ = data_filtered.query(f"{conversion_col}==0")[[touchpoints_array_col]].values
        negative_touches = [val[0] for val in negative_touchpoints_]
        distinct_touches = sorted(list(set([item for sublist in positive_touches + negative_touches for item in sublist])))
        scores = self._get_markov_scores(positive_touches, negative_touches, distinct_touches, journey_weights=conversion_weights, attribution_reports_folder_path=attribution_reports_folder_path, enable_visualisation=enable_visualisation)
        markov_df = pd.DataFrame.from_dict(scores, orient="index", columns=["markov_scores"]).reset_index()
        markov_df.columns = [touchpoints_array_col, "markov_scores"]
        return markov_df
    
    

class AttributionModelRecipe(PyNativeRecipe):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.logger = Logger("attribution_model") # create a logger for debug/error logging
        self.inputs = {
                        "var_table": f'{self.config["entity"]}/all/var_table',
                       }
        for key in ("touchpoint_var", "conversion_entity_var", "days_since_first_seen_var"):
            if key in self.config:
                self.inputs[key] = f'entity/{self.config["entity"]}/{self.config[key]}'

    # describe the compile output of the recipe - the sql code and the extension of the file
    def describe(self, this: WhtMaterial):
        #return self.python, ".py"
        description = """You can see the output table in the warehouse where each touchpoint has an attribution score according to different attribution model.\n
        Linear and Markov models are multi-touch attribution models where the conversion is attributed to multiple touchpoints. 
        If the conversion column is boolean, it gets a weight of 1. If it's a numeric column, it gets the value of the conversion.
        """
        return  description, ".txt"

    # prepare the material for execution - de_ref the inputs and create the sql code
    def register_dependencies(self, this: WhtMaterial):
        for key in self.inputs:
            this.de_ref(self.inputs[key])
    
    def _get_first_touch_scores(self, input_df: pd.DataFrame,  touchpoints_array_col: str, conversion_col:str):
        input_df = input_df.copy()
        input_df["first_touch_tmp"] = input_df[touchpoints_array_col].apply(lambda touchpoints: touchpoints[0] if touchpoints and len(touchpoints) else None)
        input_df["last_touch_tmp"] = input_df[touchpoints_array_col].apply(lambda touchpoints: touchpoints[-1] if touchpoints and len(touchpoints) else None)
        try:
            first_touch_data = (input_df
                                .groupby("first_touch_tmp")[conversion_col]
                                .sum()
                                .reset_index()
                                .filter(["first_touch_tmp", conversion_col])
                                )
            last_touch_data = (input_df
                            .groupby("last_touch_tmp")[conversion_col]
                            .sum()
                            .reset_index()
                            .filter(["last_touch_tmp", conversion_col])
                            )
        except:
            raise ValueError("conversion_entity_var is not numeric column. Please provide a numeric column for conversion.")
        first_touch_data.columns = [touchpoints_array_col, "first_touch_conversion"]
        last_touch_data.columns = [touchpoints_array_col, "last_touch_conversion"]
        del input_df["first_touch_tmp"], input_df["last_touch_tmp"]
        return pd.merge(first_touch_data, last_touch_data, on=touchpoints_array_col, how="outer")
    
    def _plot_results(self, final_scores:pd.DataFrame, touchpoint_column: str, output_folder:str):
        df_long = pd.melt(final_scores, touchpoint_column, list(final_scores))
        df_long.columns = ['touch', 'method', 'attribution']
        plt.figure(figsize=(16,6))
        sns.barplot(data=df_long, x='touch', y='attribution', hue='method', palette=sns.color_palette('Paired'), saturation=0.75)
        plt.xticks(rotation=90)
        plt.legend(loc='upper right')
        plt.xlabel('Touch')
        plt.ylabel('Attribution')
        plt.title('Attribution by Touch and Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "attribution_scores.png"))

    # execute the material
    def execute(self, this: WhtMaterial):
        multitouch_models = MultiTouchModels(self.logger)
        touch_point_var = self.config['touchpoint_var'].lower()
        enable_visualisation = self.config.get('enable_visualisation', True)
        input_df = this.de_ref(self.inputs["var_table"]).get_df()#(select_columns=[touch_point_var, conversion_var])

        if 'conversion_entity_var' in self.config:
            conversion_var = self.config['conversion_entity_var'].lower()
        else:
            conversion_var = 'is_converted'
            input_df[conversion_var] = True

        input_df.columns = [x.lower() for x in input_df.columns]

        if 'days_since_first_seen_var' in self.config:
            days_since_first_seen_var = self.config['days_since_first_seen_var'].lower()
            filtered_df = input_df.query(f"{days_since_first_seen_var} <= {self.config['first_seen_since']}").copy()
        else:
            filtered_df = input_df.copy()

        filtered_df.columns = [x.lower() for x in input_df.columns]

        def _convert_str_to_list(x):
            try:
                return x.split(",")
            except:
                return []
        filtered_df[touch_point_var] = filtered_df[touch_point_var].apply(_convert_str_to_list)

        #Create a directory with the material name in the output folder
        output_folder = this.get_output_folder() # where the output files are present
        material_name = this.name() # name of the material
        attribution_reports_folder = os.path.join(output_folder, material_name)
        
        pathlib.Path(attribution_reports_folder).mkdir(parents=True, exist_ok=True)
        attribution_scores = self._get_first_touch_scores(filtered_df,
                                                         touch_point_var, 
                                                         conversion_var)
        if enable_visualisation and (attribution_scores.shape[0] > MAXIMUM_TOUCHPOINTS_TO_VISUALIZE):
            enable_visualisation = False
            self.logger.info(f"Skipping visualising the attribution model outputs as there are too many touchpoints. Visualisation is supported only when we have fewer than {MAXIMUM_TOUCHPOINTS_TO_VISUALIZE} touchpoints.")
        linear_scores = multitouch_models.linear_model(filtered_df, touch_point_var, conversion_var)
        markov_scores = multitouch_models.get_markov_attribution(filtered_df, conversion_var, touch_point_var, attribution_reports_folder, enable_visualisation)
        attribution_scores = pd.merge(attribution_scores, linear_scores, on=touch_point_var, how="outer")
        attribution_scores = pd.merge(attribution_scores, markov_scores, on=touch_point_var, how="outer")
        try:
            self._plot_results(attribution_scores, touch_point_var, attribution_reports_folder)
        except Exception as e:
            self.logger.error(f"Could not plot the attribution scores: {e}")

        total_conversions = attribution_scores['first_touch_conversion'].sum()
        for col in list(attribution_scores):
            if col not in [touch_point_var, 'total_conversions']:
                attribution_scores[f"{col}_normalised"] = 100*attribution_scores[col].div(total_conversions, axis=0).round(2)

        attribution_scores = attribution_scores.sort_values(by='first_touch_conversion', ascending=False)
        
        this.write_output(attribution_scores)
