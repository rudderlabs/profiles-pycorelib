# Commenting out this model as this has some breaking changes for v0.10.4
# This will be fixed in the next release
from typing import List, Dict, Tuple, Union

from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.schema import ContractBuildSpecSchema, EntityKeyBuildSpecSchema, EntityIdsBuildSpecSchema, MaterializationBuildSpecSchema
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import os
import pathlib

matplotlib.use('agg')

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
            "entity": { "type": "string"}
            # extend your schema from pre-defined schemas
            # **ContractBuildSpecSchema["properties"],
            # **EntityKeyBuildSpecSchema["properties"],
            # **EntityIdsBuildSpecSchema["properties"],
            # **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["entity", "touchpoint_var", "days_since_first_seen_var", "first_seen_since", "conversion_entity_var"],
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
    def linear_model(input_df, touchpoints_array_col, conversion_col:str) -> pd.DataFrame:
        input_df = input_df.copy()
        input_df['n_touches'] = input_df[touchpoints_array_col].apply(lambda x: len(x) if x else 0)
        input_df['weight'] = input_df.apply(lambda row: row[conversion_col]/row['n_touches'] if row['n_touches']>0 else 0, axis=1)
        linear_scores = input_df.explode(touchpoints_array_col).groupby(touchpoints_array_col).agg({'weight':'sum'}).reset_index()
        linear_scores.columns = [touchpoints_array_col, "linear_score"]
        del input_df['n_touches'], input_df['weight']
        return linear_scores
    
    @staticmethod
    def _generate_transition_counts(journey_list: List[List[Union[int, str]]], 
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
    @staticmethod
    def _plot_transitions(transition_probabilities: np.array, 
                          labels: List[Union[int, str]], 
                          image_path, title="Markov model transition probabilities", 
                          show_annotations=True):
        ax = sns.heatmap(transition_probabilities,
                        linewidths=0.5,
                        robust=True, 
                        annot_kws={"size":8}, 
                        annot=show_annotations,
                        fmt=".2f",
                        cmap="YlGnBu",
                        xticklabels=labels,
                        yticklabels=labels)
        ax.tick_params(labelsize=10)
        ax.figure.set_size_inches((16, 10))
        ax.set_ylabel("Previous Step")
        ax.set_xlabel("Next Step")
        ax.set_title(title)
        plt.savefig(image_path)
    
    @staticmethod
    def _get_transition_probabilities(converted_touchpoints_list: List[List[Union[int, str]]], 
                                    dropoff_touchpoints_list: List[List[Union[int, str]]], 
                                    distinct_touches_list: List[Union[int, str]],
                                    journey_weights: List[float] = None) -> Tuple[np.array, List[Union[int, str]]]:
        row_normalize_np_array = lambda transition_counts: np.where(transition_counts.sum(axis=1)[:, np.newaxis] != 0,
                                                            transition_counts / transition_counts.sum(axis=1)[:, np.newaxis],
                                                            0)
        pos_transitions, _ = MultiTouchModels._generate_transition_counts(converted_touchpoints_list, distinct_touches_list, is_positive=True, journey_weights=journey_weights)
        neg_transitions, labels = MultiTouchModels._generate_transition_counts(dropoff_touchpoints_list, distinct_touches_list, is_positive=False)
        all_transitions = pos_transitions + neg_transitions
        transition_probabilities = row_normalize_np_array(all_transitions)
        return transition_probabilities, labels
    @staticmethod
    def _converge(transition_matrix, max_iters=200, verbose=True):
        T_upd = transition_matrix
        prev_T = transition_matrix
        for i in range(max_iters):
            T_upd = np.matmul(transition_matrix, prev_T)
            if np.abs(T_upd - prev_T).max()<1e-5:
                if verbose:
                    print(f"{i} iters taken for convergence")
                return T_upd
            prev_T = T_upd
        if verbose:
            print(f"Max iters of {max_iters} reached before convergence. Exiting")
        return T_upd
    
    @staticmethod
    def _get_removal_affects(transition_probs, labels, ignore_labels=["Start", "Dropoff","Converted"], default_conversion=1.):
        removal_affect = {}
        for n, label in enumerate(labels):
            if label in ignore_labels:
                continue
            else:
                drop_transition = transition_probs.copy()
                drop_transition[n,:] = 0. # Drop all transitions from this touchpoint
                drop_transition[n,-2] = 1. # Force all touches to dropoff from this touchpoint
                drop_transition_converged = MultiTouchModels._converge(drop_transition, 500, False)
                removal_affect[label] = default_conversion - drop_transition_converged[0,-1]
        return removal_affect
    
    @staticmethod
    def _get_markov_scores(tp_list_positive: List[List[Union[int, str]]],
                            tp_list_negative: List[List[Union[int, str]]], 
                            distinct_touches_list: List[str],
                            journey_weights: List[float]=None,
                            attribution_reports_folder_path:str=None) -> Tuple[Dict[Union[int, str], float], np.array]:
        transition_probabilities, labels = MultiTouchModels._get_transition_probabilities(tp_list_positive, tp_list_negative, distinct_touches_list, journey_weights=journey_weights)
        transition_probabilities_converged = MultiTouchModels._converge(transition_probabilities, max_iters=500, verbose=False)
        if attribution_reports_folder_path:
            image_file = os.path.join(attribution_reports_folder_path, "markov_transition_probabilities.png")
            MultiTouchModels._plot_transitions(transition_probabilities, labels, image_file)
        removal_affects = MultiTouchModels._get_removal_affects(transition_probabilities, labels, default_conversion=transition_probabilities_converged[0,-1])
        total_conversions = sum(journey_weights) if journey_weights else len(tp_list_positive)
        attributable_conversions = {}
        total_weight = sum(removal_affects.values())
        for tp, weight in removal_affects.items():
            attributable_conversions[tp] = weight/total_weight * total_conversions
        return attributable_conversions
    
    @staticmethod
    def get_markov_attribution(input_df: pd.DataFrame, conversion_col: str, touchpoints_array_col: str, attribution_reports_folder_path: str) -> pd.DataFrame:
        data_filtered = input_df[input_df[touchpoints_array_col].apply(lambda touches: len(touches)>0 if touches else False)]
        positive_touchpoints_ = data_filtered.query(f"{conversion_col}>0")[[touchpoints_array_col, conversion_col]].values
        positive_touches = [val[0] for val in positive_touchpoints_]
        conversion_weights = [val[1] for val in positive_touchpoints_]
        negative_touchpoints_ = data_filtered.query(f"{conversion_col}==0")[[touchpoints_array_col]].values
        negative_touches = [val[0] for val in negative_touchpoints_]
        distinct_touches = sorted(list(set([item for sublist in positive_touches + negative_touches for item in sublist])))
        scores = MultiTouchModels._get_markov_scores(positive_touches, negative_touches, distinct_touches, journey_weights=conversion_weights, attribution_reports_folder_path=attribution_reports_folder_path)
        markov_df = pd.DataFrame.from_dict(scores, orient="index", columns=["markov_scores"]).reset_index()
        markov_df.columns = [touchpoints_array_col, "markov_scores"]
        return markov_df
    
    

class AttributionModelRecipe(PyNativeRecipe):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.logger = Logger("attribution_model") # create a logger for debug/error logging
        self.inputs = {"touchpoints": f'entity/{self.config["entity"]}/{self.config["touchpoint_var"]}',
                       "conversion": f'entity/{self.config["entity"]}/{self.config["conversion_entity_var"]}', 
                       "days_since_first_seen": f'entity/{self.config["entity"]}/{self.config["days_since_first_seen_var"]}'
                       }


    # describe the compile output of the recipe - the sql code and the extension of the file
    def describe(self, this: WhtMaterial):
        #return self.python, ".py"
        description = """You can see the output table in the warehouse where each touchpoint has an attribution score according to different attribution model.\n
        Linear and Markov models are multi-touch attribution models where the conversion is attributed to multiple touchpoints. 
        If the conversion column is boolean, it gets a weight of 1. If it's a numeric column, it gets the value of the conversion - except for in markov models.
        In markov models, each conversion is treated equally (for now). 
        So if the conversion column is numeric, it is likely that in a markov model, the scores will look very different compared to other models.
        """
        return  description, ".txt"

    # prepare the material for execution - de_ref the inputs and create the sql code
    def prepare(self, this: WhtMaterial):
        this.de_ref(self.inputs["touchpoints"])
        this.de_ref(self.inputs["days_since_first_seen"])
        this.de_ref(self.inputs["conversion"])
        #return "foo", ".txt"
    
    
    def _get_first_touch_scores(self, input_df: pd.DataFrame,  touchpoints_array_col: str, conversion_col:str):
        input_df = input_df.copy()
        input_df["first_touch_tmp"] = input_df[touchpoints_array_col].apply(lambda touchpoints: touchpoints[0] if touchpoints and len(touchpoints) else None)
        input_df["last_touch_tmp"] = input_df[touchpoints_array_col].apply(lambda touchpoints: touchpoints[-1] if touchpoints and len(touchpoints) else None)
        first_touch_data = (input_df
                            .groupby("first_touch_tmp")
                            .sum(conversion_col)
                            .reset_index()
                            .filter(["first_touch_tmp", conversion_col])
                            )
        last_touch_data = (input_df
                        .groupby("last_touch_tmp")
                        .sum(conversion_col)
                        .reset_index()
                        .filter(["last_touch_tmp", conversion_col])
                        )
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
        touch_point_var = self.config['touchpoint_var'].lower()
        conversion_var = self.config['conversion_entity_var'].lower()
        days_since_first_seen_var = self.config['days_since_first_seen_var'].lower()
        input_df = this.de_ref(f'{self.config["entity"]}/all/var_table').get_df()#(select_columns=[touch_point_var, conversion_var])
        input_df.columns = [x.lower() for x in input_df.columns]
        filtered_df = input_df.query(f"{days_since_first_seen_var} <= {self.config['first_seen_since']}").copy()
        
        filtered_df.columns = [x.lower() for x in input_df.columns]
        def _convert_str_to_list(x):
            try:
                return ast.literal_eval(x)
            except ValueError:
                return []
        filtered_df[touch_point_var] = filtered_df[touch_point_var].apply(_convert_str_to_list)
        # user_var_table, 
        #touchpoint_data.join(conversion_data, on="user_id", how="outer")
        
        #Create a directory with the material name in the output folder
        output_folder = this.get_output_folder() # where the output files are present
        material_name = this.name() # name of the material
        attribution_reports_folder = os.path.join(output_folder, material_name)
        
        pathlib.Path(attribution_reports_folder).mkdir(parents=True, exist_ok=True)
        attribution_scores = self._get_first_touch_scores(filtered_df,
                                                         touch_point_var, 
                                                         conversion_var)
        linear_scores = MultiTouchModels.linear_model(filtered_df, touch_point_var, conversion_var)
        markov_scores = MultiTouchModels.get_markov_attribution(filtered_df, conversion_var, touch_point_var, attribution_reports_folder)
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
