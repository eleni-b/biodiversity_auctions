#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation

# In[20]:


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import json
from bisect import bisect
from abc import *
from enum import Enum
import gurobipy as gp
from gurobipy import GRB
import random


## Utils

def powerset_by_length(items, min_len=1, max_len_exclusive=None, unpack_single=True):
    """
    Generate the subsets of elements in the iterable, in order of length.

    Args:
        items (iterable): The input iterable from which subsets are generated.
        min_len (int): Minimum length of the subsets (inclusive). Defaults to 1.
        max_len_exclusive (int): Maximum length of the subsets (exclusive). Defaults to len(items)+1.
        unpack_single (bool): If True, unpack single-element subsets. Defaults to True.

    Yields:
        tuple or element: Subsets of the input iterable.
    """
    if max_len_exclusive is None:
        max_len_exclusive = len(items) + 1

    for k in range(min_len, max_len_exclusive):
        for subset in itertools.combinations(items, k):
            yield subset[0] if k == 1 and unpack_single else subset

def values_key(tup_key):
    """
    Generate a key for a subset of items.

    Args:
        tup_key (tuple): The subset of items.

    Returns:
        tuple or element: Key for the subset.
    """
    return tup_key[0] if len(tup_key) == 1 else tup_key


## Class Definitions


class AuctionItem(ABC):
    """
    Abstract base class representing an auction item.
    """
    @property
    @abstractmethod
    def identifier(self):
        pass

    @property
    @abstractmethod
    def cost(self) -> float:
        pass


class Land(AuctionItem):
    """
    Concrete class representing a piece of land in the auction.

    Attributes:
        identifier (tuple[int, int]): The identifier of the land.
        cost (int): The cost of the land.
    """
    def __init__(self, identifier, cost):
        self.identifier = identifier
        self.cost = cost
        

class ValuationFunction(ABC):
    """
    Abstract base class for valuation functions.
    """
    @abstractmethod
    def generate_valuations(self, auction, mult):
        vals = {}
        for item in auction.items:
            val = np.random.randint(0, auction.value_max)
            while val < auction.costs[item]:
                val = np.random.randint(0, auction.value_max)
            vals[item] = val
        return vals


class AdditiveValuation(ValuationFunction):
    """
    Additive valuation function.
    """
    def generate_valuations(self, auction, mult):
        values = super().generate_valuations(auction, mult)
        for subset in auction.itemgroups_to_value:
            if subset in values:
                continue
            values[subset] = sum(values[s] for s in subset)
        return values


class SuperadditiveValuation(ValuationFunction):
    """
    Superadditive valuation function.
    """
    def generate_valuations(self, auction, mult):
        values = super().generate_valuations(auction, mult)
        
        # Generate valuations for all subsets
        for subset in auction.itemgroups_to_value:
            if subset not in values:
                values[subset] = sum(values[s] for s in subset)
        
        # Apply multiplier to non-singleton subsets
        for subset in auction.itemgroups_to_value:
            if len(subset) > 1:
                values[subset] *= mult
        
        return values


class SubmodularNonmonotoneValuation(ValuationFunction):
    """
    Submodular non-monotone valuation function.
    """
    def generate_valuations(self, auction, mult):
        values = super().generate_valuations(auction, mult)
        for subset in auction.itemgroups_to_value:
            if subset in values:
                continue
            max_val = min(
                values[values_key(subsubset)] + values[values_key(subsubset2)] - 
                values.get(values_key(tuple(sorted(set(subsubset) & set(subsubset2)))), 0)
                for subsubset in powerset_by_length(subset, len(subset)//2, len(subset), False)
                for subsubset2 in powerset_by_length(subset, 1, len(subset), False)
                if set(subsubset) | set(subsubset2) == set(subset)
            )
            values[subset] = np.random.randint(max_val * 0.9, max_val)
        return values


class SubmodularMonotoneValuation(ValuationFunction):
    """
    Submodular monotone valuation function.
    """
    def generate_valuations(self, auction, mult):
        values = super().generate_valuations(auction, mult)
        for subset in auction.itemgroups_to_value:
            if subset in values:
                continue
            
            max_val = float('inf')
            min_val = 0

            # Calculate max_val and min_val in a single loop to optimize performance
            for subsubset in powerset_by_length(subset, 1, len(subset), False):
                if len(subsubset) == len(subset) // 2:
                    for subsubset2 in powerset_by_length(subset, 1, len(subset), False):
                        if set(subsubset) | set(subsubset2) == set(subset):
                            intersection_value = values.get(values_key(tuple(sorted(set(subsubset) & set(subsubset2)))), 0)
                            combined_value = values[values_key(subsubset)] + values[values_key(subsubset2)] - intersection_value
                            max_val = min(max_val, combined_value)
                
                # Calculate min_val
                min_val = max(min_val, values[values_key(subsubset)])

            # Set the value for the subset
            values[subset] = max(min_val, max_val - np.random.randint(0, max(1, min(int(max_val * 0.25), int((max_val - min_val) * 0.35)))))

class Auction:
    """
    A class to represent an auction.

    Attributes:
        valuation_function (ValuationFunction): The valuation function used in the auction.
        item_objs (list[AuctionItem]): A list of auction item objects.
        items (list[str]): A list of item identifiers.
        costs (dict): A dictionary mapping item identifiers to their costs.
        budget (int): The budget for the auction.
        multiplier (float): A multiplier used for valuation generation.
        itemgroups_to_value (list): A list of item groups to be valued.
        values (dict): A dictionary mapping item groups to their values.
        corridor_value (dict): A dictionary to store corridor values.
    """

    costs_max = 50
    value_max = 100

    def __init__(self, valuation_function, items, budget, multiplier):
        """
        Initialize the Auction with a valuation function, items, budget, and multiplier.
        
        Args:
            valuation_function (ValuationFunction): The valuation function used in the auction.
            items (list[AuctionItem]): A list of auction item objects.
            budget (int): The budget for the auction.
            multiplier (float): A multiplier used for valuation generation.
        """
        self.valuation_function = valuation_function
        self.item_objs = items
        self.items = [item.identifier for item in items]
        self.costs = {item.identifier: item.cost for item in self.item_objs}
        self.itemgroups_to_value = self.items + list(powerset_by_length(self.items, 2))
        self.values = self.valuation_function.generate_valuations(self, multiplier)
        self.corridor_value = {}
        self.budget = budget
        self.multiplier = multiplier

        self._calculate_corridor_values()

    def _calculate_corridor_values(self):
        """
        Calculate the corridor values for item groups.
        """
        for item_set in self.itemgroups_to_value:
            corridor = self._check_corridor(item_set)
            if corridor == 1:
                self._update_values_for_corridor(item_set, self.multiplier)
            elif corridor == 2:
                self._update_values_for_double_corridor(item_set)
            elif corridor > 2:
                self._update_values_for_multiple_corridors(item_set)
        
        self._finalize_values()

    def _check_corridor(self, item_set):
        """
        Check the corridor condition for the given item set.
        
        Args:
            item_set (tuple): The item set to check.
        
        Returns:
            int: The corridor level.
        """
        list_i = [item[0] for item in item_set if isinstance(item, tuple)]
        check = sum(1 for i in list_i if list_i.count(i) == 3)
        return check // 3 if check > 0 else 0

    def _update_values_for_corridor(self, item_set, multiplier):
        """
        Update the values for item sets with corridor level 1.
        
        Args:
            item_set (tuple): The item set to update.
            multiplier (float): The multiplier for updating values.
        """
        self.values[item_set] *= multiplier
        self.corridor_value[item_set] = self.values[item_set]

    def _update_values_for_double_corridor(self, item_set):
        """
        Update the values for item sets with corridor level 2.
        
        Args:
            item_set (tuple): The item set to update.
        """
        value_sum, count = 0, 0
        for subset in self.corridor_value:
            if all(item in item_set for item in subset):
                value_sum += self.values[subset]
                count += 1
        self.values[item_set] = value_sum
        self.corridor_value[item_set] = self.values[item_set]

    def _update_values_for_multiple_corridors(self, item_set):
        """
        Update the values for item sets with corridor level greater than 2.
        
        Args:
            item_set (tuple): The item set to update.
        """
        min_value = float('inf')
        for this_item in self.corridor_value:
            if len(this_item) > 3:
                for that_item in self.corridor_value:
                    if len(that_item) > 3 and set(this_item) | set(that_item) == set(item_set):
                        intersect_value = self.values.get(self._values_key(tuple(sorted(set(this_item) & set(that_item)))), 0)
                        combined_value = self.values[this_item] + self.values[that_item] - intersect_value
                        min_value = min(min_value, combined_value)
        self.values[item_set] = min_value
        self.corridor_value[item_set] = self.values[item_set]

    def _finalize_values(self):
        """
        Finalize the values for item sets not in the corridor values.
        """
        for item_set in self.itemgroups_to_value:
            if item_set not in self.corridor_value:
                corridor_check, corridor_indices = self._get_corridor_indices(item_set)
                for index in set(corridor_indices):
                    self.values[item_set] += (self.values[(index, 1)] + self.values[(index, 2)] + self.values[(index, 3)]) * (self.multiplier - 1)

    def _get_corridor_indices(self, item_set):
        """
        Get the corridor indices for the given item set.
        
        Args:
            item_set (tuple): The item set to check.
        
        Returns:
            tuple: A tuple containing the corridor check value and list of corridor indices.
        """
        list_i = [item[0] for item in item_set if isinstance(item, tuple)]
        corridor_check = sum(1 for i in list_i if list_i.count(i) == 3)
        corridor_indices = [i for i in list_i if list_i.count(i) == 3]
        return corridor_check, corridor_indices

    @staticmethod
    def _values_key(subset):
        """
        Generate a key for a subset of items.
        
        Args:
            subset (tuple): The subset of items.
        
        Returns:
            tuple: A sorted tuple representing the key.
        """
        return tuple(sorted(subset))




class Mechanism(ABC):
    
    @staticmethod
    def _add_indicators(model, x, auction):
        """
        Adds indicator variables to the model for each item in the auction.

        Args:
            model: Gurobi model.
            x: Dictionary of binary variables representing item groups.
            auction: Auction object containing items and their values.

        Returns:
            obj_exp: List of objective expressions.
        """
        obj_exp = []
        for item in auction.itemgroups_to_value:
            dx = [0, 0, 1, -1]
            dy = [1, -1, 0, 0]
            neighbors = []
            l1, l2 = item
            for i, j in zip(dx, dy):
                if (l1 + i, l2 + j) in x:
                    neighbors.append(x[l1 + i, l2 + j])
            indicator_vars = []
            for n in range(len(neighbors) + 1):
                indicator_var = model.binary_var(name=f'indicator_{item}_{n}')
                indicator_vars.append(indicator_var)
                obj_exp.append(x[item] * auction.benefits[item][n] * indicator_var)
                model.add_indicator(indicator_var, model.sum(neighbors) == n, name=f'indicator_{n}_for_{item}')
            model.add_constraint(model.sum(indicator_vars) == 1, name=f'indicator_single_assign_{item}')
        return obj_exp
    
    @staticmethod
    def _calculate_benefit(auction, x):
        """
        Calculates the total benefit of selected items.

        Args:
            auction: Auction object containing items and their values.
            x: List of binary variables indicating selected items.

        Returns:
            Total benefit of selected items.
        """
        pos_vars = tuple([item for item, x_i in zip(auction.items, x) if x_i])
        return auction.values.get(pos_vars[0] if len(pos_vars) == 1 else pos_vars, 0)

    @staticmethod
    def _calculate_cost(auction, x):
        """
        Calculates the total cost of selected items.

        Args:
            auction: Auction object containing items and their costs.
            x: List of binary variables indicating selected items.

        Returns:
            Total cost of selected items.
        """
        return sum(auction.costs[auction.items[i]] for i, xi in enumerate(x) if xi)

    @abstractmethod
    def get_assignments_and_prices(self, trial_name, auction):
        """
        Abstract method to get assignments and prices for the auction.

        Args:
            trial_name: Name of the trial.
            auction: Auction object containing items and their values.

        Returns:
            Gurobi model and cost variables.
        """
        model = gp.Model(trial_name)
        model.setParam('OutputFlag', False)
        x = {item: model.addVar(vtype=GRB.BINARY, name=f"x_{item}") for item in auction.itemgroups_to_value}
        obj_exp = [auction.values[item] * x[item] for item in auction.itemgroups_to_value]
        model.addConstr(gp.quicksum(x[item] for item in auction.itemgroups_to_value) <= 1, name='constr1')

        y = {item: model.addVar(vtype=GRB.BINARY, name=f"y_{item}") for item in auction.items}
        for item_set, x_item in x.items():
            if not isinstance(item_set[0], tuple):
                item_set = (item_set,)
            model.addConstr((x_item == 1) >> (gp.quicksum(y[item] for item in item_set) == len(item_set)), name=f'constr2_{item_set}')
            model.addConstr((x_item == 0) >> (gp.quicksum(y[item] for item in item_set) <= len(item_set)), name=f'constr3_{item_set}')
        
        cost_var = y
        costs_exp = model.addVar(vtype=GRB.INTEGER, name='c')
        model.addConstr(costs_exp == gp.quicksum(cost_var[k] * cost for k, cost in auction.costs.items()), name='constr4')
        model.setObjective(gp.quicksum(obj_exp) - costs_exp, GRB.MAXIMIZE)

        return model, cost_var


class VCGMechanism(Mechanism):
    
    @staticmethod
    def _optimal_with_constraint(model, constraint, constraint_name):
        """
        Optimizes the model with a given constraint.

        Args:
            model: Gurobi model.
            constraint: Constraint to add to the model.
            constraint_name: Name of the constraint.

        Returns:
            Net benefits of the model after optimization.
        """
        temp_constr = model.addConstr(constraint, name=constraint_name)
        model.optimize()
        model.write('vcg.lp')
        net_benefits = model.objVal if model.status == GRB.OPTIMAL else None
        model.remove(temp_constr)
        return net_benefits

    def get_assignments_and_prices(self, trial_name, auction):
        """
        Gets assignments and prices for the VCG mechanism.

        Args:
            trial_name: Name of the trial.
            auction: Auction object containing items and their values.

        Returns:
            Net benefits and total payments.
        """
        model, y = super().get_assignments_and_prices(trial_name, auction)
        total_payment = 0
        assignments = []

        for item, item_var in y.items():
            conserve_net_benefits = VCGMechanism._optimal_with_constraint(model, item_var == 1, f'{item}_conserve')
            develop_net_benefits = VCGMechanism._optimal_with_constraint(model, item_var == 0, f'{item}_develop')

            if conserve_net_benefits is None or develop_net_benefits is None:
                print(f'Error in iteration for item {item}')
                assignments.append(0)
                continue

            conserve_net_benefits += auction.costs[item]
            payment = conserve_net_benefits - develop_net_benefits

            if payment >= auction.costs[item]:
                total_payment += payment
                print(f'Landowner of {item} will be paid {round(payment, 2)}, had bid/cost of {auction.costs[item]}')
                assignments.append(1)
            else:
                assignments.append(0)

        net_benefits = Mechanism._calculate_benefit(auction, assignments) - Mechanism._calculate_cost(auction, assignments)
        return net_benefits, total_payment


class OptimalBudgetMechanism(Mechanism):
    
    def get_assignments_and_prices(self, trial_name, auction):
        """
        Gets assignments and prices for the optimal budget mechanism.

        Args:
            trial_name: Name of the trial.
            auction: Auction object containing items and their values.

        Returns:
            Objective value and total payments.
        """
        model, y = super().get_assignments_and_prices(trial_name, auction)
        price = {item: model.addVar(vtype=GRB.INTEGER, name=f"p_{item}") for item in y.keys()}

        model.addConstrs((price[yi] >= y[yi] * auction.costs[yi] for yi in y), name='constr5')
        model.addConstr(gp.quicksum(price.values()) <= auction.budget, name='budget')
        model.optimize()
        model.write('opt.lp')

        objective = model.objVal
        payments = sum(v.X for v in price.values())

        for v in model.getVars():
            if v.X != 0:
                print(f"{v.varName}: {v.X}")

        return objective, payments


class ClockBudgetMonotoneMechanism(Mechanism):

    def get_assignments_and_prices(self, trial_name, auction):
        """
        Gets assignments and prices for the clock budget monotone mechanism.

        Args:
            trial_name: Name of the trial.
            auction: Auction object containing items and their values.

        Returns:
            Net benefits and total payments.
        """
        active_bidders = dict.fromkeys(auction.items)
        prev_accept_bidders = dict()
        cur_accept_bidders = {auction.items[np.argmax([auction.values[item] for item in auction.items])]: True}
        cur_round = 1
        prices = {b: auction.budget for b in active_bidders}
        cur_optim_estimate = self._get_set_value(auction.values, cur_accept_bidders)

        while (remain_bidders := self._dict_keys_difference(active_bidders, prev_accept_bidders | cur_accept_bidders)):
            cur_round += 1
            cur_optim_estimate *= 2
            prev_accept_bidders = dict(cur_accept_bidders)
            cur_accept_bidders = dict()

            while self._get_set_value(auction.values, cur_accept_bidders) < cur_optim_estimate and remain_bidders:
                max_marginal_contrib = max(remain_bidders, key=lambda bidder: self._get_set_value(auction.values, cur_accept_bidders | {bidder: True}, marginal=True))
                prices[max_marginal_contrib] = min(prices[max_marginal_contrib], round(self._get_set_value(auction.values, cur_accept_bidders | {max_marginal_contrib: True}, marginal=True) * auction.budget / cur_optim_estimate, 2))

                if auction.costs[max_marginal_contrib] <= prices[max_marginal_contrib]:
                    cur_accept_bidders[max_marginal_contrib] = True
                else:
                    active_bidders.pop(max_marginal_contrib)

        w1 = dict(prev_accept_bidders)
        w2_ = dict(cur_accept_bidders)

        if sum(prices[i] for i in w1) > auction.budget:
            last_added_bidder = list(prev_accept_bidders)[-1]
            prices[last_added_bidder] = min(prices[last_added_bidder], round(self._get_set_value(auction.values, cur_accept_bidders | {last_added_bidder: None}, marginal=True) * auction.budget / cur_optim_estimate, 2))
            w1.pop(last_added_bidder)
            if auction.costs[last_added_bidder] <= prices[last_added_bidder]:
                w2_[last_added_bidder] = True

        return self._max_value(w1, w2_, prices, auction)
    
    @staticmethod
    def _get_set_value(benefits, bid_set, marginal=False):
        """
        Gets the value of a set of bids.

        Args:
            benefits: Dictionary of benefits.
            bid_set: Set of bids.
            marginal: Boolean indicating whether to calculate marginal value.

        Returns:
            Value of the set of bids.
        """
        bid_tup = tuple(sorted(bid_set))
        value = benefits.get(bid_tup[0] if len(bid_tup) == 1 else bid_tup, 0)
        if marginal and len(bid_set) > 1:
            value -= ClockBudgetMonotoneMechanism._get_set_value(benefits, list(bid_set)[:-1])
        return value

    @staticmethod
    def _dict_keys_difference(d1, d2):
        """
        Gets the difference between the keys of two dictionaries.

        Args:
            d1: First dictionary.
            d2: Second dictionary.

        Returns:
            Dictionary containing keys that are in d1 but not in d2.
        """
        rem_keys = set(d1) - set(d2)
        return {k: d1[k] for k in rem_keys}

    @staticmethod
    def _max_value(w1, w2_, prices, auction):
        """
        Calculates the maximum value of the selected bids.

        Args:
            w1: Dictionary of first set of selected bids.
            w2_: Dictionary of second set of selected bids.
            prices: Dictionary of prices for each bid.
            auction: Auction object containing items and their values.

        Returns:
            Tuple of maximum value and total price.
        """
        cum_prices_w2_ = np.cumsum([prices[i] for i in w2_])
        budget_idx = bisect(cum_prices_w2_, auction.budget)
        w2 = dict.fromkeys(list(w2_)[:budget_idx])

        if w2_:
            T = dict.fromkeys(list(w1)[:bisect(np.cumsum([prices[i] for i in w1]), auction.budget - cum_prices_w2_[budget_idx - 1])])
            w3 = w2 | T
        else:
            w3 = {}

        get_value_fn = ClockBudgetMonotoneMechanism._get_set_value
        w = w3 if get_value_fn(auction.values, w1) < get_value_fn(auction.values, w3) else w1

        return get_value_fn(auction.values, w) - sum(auction.costs[i] for i in w), sum(prices[i] for i in w)
    




## Experimental part

# Define main function
def run_experiments():
    testsets_per_val_type = 10
    val_types = [
        # ('additive', AdditiveValuation), 
        ('superadditive', SuperadditiveValuation), 
        # ('submodular_monotone', SubmodularMonotoneValuation),
        # ('submodular_nonmonotone', SubmodularNonmonotoneValuation)
    ]

    budget_list = random.sample(range(70, 80), 10)
    np.random.seed(11)
    results = pd.DataFrame(columns=['binding', 'budget', 'clock_mono_obj', 'clock_mono_payment', 'size', 'valuation_fn', 'multiplier', 'clock_mono_runtime'])

    # Grid size
    rows, cols = 3, 3

    # Superadditivity multiplier range
    superadditive_multiplier = [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
    multiplier = superadditive_multiplier[6]

    for vtype, vtype_fn in val_types:
        valuation_function = vtype_fn()
        for ti in range(testsets_per_val_type):
            auction_items = [Land((i + 1, j + 1), np.random.randint(1, 50)) for i in range(rows) for j in range(cols)]
            auction = Auction(valuation_function, auction_items, budget_list[ti], multiplier)
            print(f'Testset {ti} for val {vtype} of rows={rows} cols={cols}')
            print(f'budget {auction.budget} - multiplier {auction.multiplier}')

            vcg_runtime = datetime.now()
            vcg_obj_val, vcg_payment = VCGMechanism().get_assignments_and_prices('VCG', auction)
            vcg_runtime = datetime.now() - vcg_runtime

            opt_runtime = datetime.now()
            budg_obj_val, budg_payment = OptimalBudgetMechanism().get_assignments_and_prices('optimal budget', auction)
            opt_runtime = datetime.now() - opt_runtime

            clock_runtime = datetime.now()
            clock_mono_obj_val, clock_mono_payment = ClockBudgetMonotoneMechanism().get_assignments_and_prices('clock monotone', auction)
            clock_mono_runtime = datetime.now() - clock_runtime

            results.loc[len(results)] = [
                'yes', auction.budget, clock_mono_obj_val, clock_mono_payment,
                (rows, cols), vtype, multiplier, clock_mono_runtime
            ]

    results = results.sort_values(by=['budget']).reset_index(drop=True)
    corridor_file = f'corridor_results_superadd-{multiplier}.csv'
    results.to_csv(corridor_file)
    print(f'Results saved to {corridor_file}')

if __name__ == "__main__":
    run_experiments()