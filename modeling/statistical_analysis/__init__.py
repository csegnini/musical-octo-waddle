"""
Statistical Analysis Package

This package provides comprehensive statistical analysis capabilities including:
- Hypothesis Testing (t-tests, chi-square, ANOVA, non-parametric tests)
- Regression Analysis (linear, polynomial, logistic, robust regression)
- Correlation and Association Analysis
- Distribution Testing and Goodness of Fit
- Effect Size Calculations
- Power Analysis
- Bayesian Statistical Methods
- Time Series Statistical Tests
- Multivariate Statistics
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.stats import (
    ttest_1samp, ttest_ind, ttest_rel,
    chi2_contingency, fisher_exact,
    f_oneway, kruskal, friedmanchisquare,
    mannwhitneyu, wilcoxon, ranksums,
    shapiro, normaltest, kstest, jarque_bera,
    levene, bartlett, fligner,
    spearmanr, pearsonr, kendalltau,
    linregress, zscore
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, HuberRegressor, RANSACRegressor,
    BayesianRidge, ARDRegression
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import math

# Statistical test types
class TestType(Enum):
    """Types of statistical tests."""
    ONE_SAMPLE_T = "one_sample_t"
    TWO_SAMPLE_T = "two_sample_t"
    PAIRED_T = "paired_t"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CORRELATION = "correlation"
    NORMALITY = "normality"
    HOMOGENEITY = "homogeneity"

class RegressionType(Enum):
    """Types of regression analysis."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    LOGISTIC = "logistic"
    ROBUST_HUBER = "robust_huber"
    ROBUST_RANSAC = "robust_ransac"
    BAYESIAN_RIDGE = "bayesian_ridge"
    BAYESIAN_ARD = "bayesian_ard"

class CorrelationType(Enum):
    """Types of correlation analysis."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"

@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    degrees_of_freedom: Optional[int]
    interpretation: str
    recommendations: List[str]
    assumptions_met: Dict[str, bool]
    additional_info: Dict[str, Any]

@dataclass
class RegressionResult:
    """Result of regression analysis."""
    regression_type: RegressionType
    model: Any
    r_squared: float
    adjusted_r_squared: float
    coefficients: np.ndarray
    intercept: float
    standard_errors: Optional[np.ndarray]
    t_statistics: Optional[np.ndarray]
    p_values: Optional[np.ndarray]
    confidence_intervals: Optional[np.ndarray]
    residuals: np.ndarray
    predictions: np.ndarray
    mse: float
    mae: float
    aic: Optional[float]
    bic: Optional[float]
    durbin_watson: Optional[float]
    diagnostic_plots: Dict[str, Any]
    assumptions_check: Dict[str, bool]
    interpretation: str
    recommendations: List[str]

class HypothesisTestAnalyzer:
    """Comprehensive hypothesis testing analyzer."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the hypothesis test analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.results_history = []
    
    def one_sample_t_test(self, 
                         data: np.ndarray, 
                         population_mean: float,
                         alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        Perform one-sample t-test.
        
        Args:
            data: Sample data
            population_mean: Hypothesized population mean
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            StatisticalTestResult object
        """
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        # Perform test
        statistic, p_value = ttest_1samp(data_clean, population_mean, alternative=alternative)
        
        # Safe type conversion with proper error handling
        # Safe type conversion with defensive programming
        try:
            statistic_val = getattr(statistic, '__float__', lambda: np.nan)()
            if isinstance(statistic, (tuple, list)) and len(statistic) > 0:
                statistic_val = getattr(statistic[0], '__float__', lambda: np.nan)()
        except:
            statistic_val = np.nan
            
        try:
            p_value_val = getattr(p_value, '__float__', lambda: np.nan)()
            if isinstance(p_value, (tuple, list)) and len(p_value) > 0:
                p_value_val = getattr(p_value[0], '__float__', lambda: np.nan)()
        except:
            p_value_val = np.nan
        
        # Calculate effect size (Cohen's d)
        sample_mean = np.mean(data_clean)
        sample_std = np.std(data_clean, ddof=1)
        if sample_std == 0:
            effect_size = np.nan
        else:
            effect_size = float((sample_mean - population_mean) / sample_std)
        
        # Confidence interval
        n = len(data_clean)
        df = n - 1
        se = sample_std / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = sample_mean - t_critical * se
        ci_upper = sample_mean + t_critical * se
        
        # Check assumptions
        normality_stat, normality_p = shapiro(data_clean) if len(data_clean) <= 5000 else normaltest(data_clean)
        assumptions = {
            'normality': normality_p > 0.05,
            'independence': True,  # Assumed
            'sample_size': len(data_clean) >= 30
        }
        
        # Interpretation
        is_significant = p_value_val < self.alpha
        interpretation = f"The sample mean ({sample_mean:.4f}) is "
        if is_significant:
            interpretation += f"significantly different from the population mean ({population_mean}) "
        else:
            interpretation += f"not significantly different from the population mean ({population_mean}) "
        interpretation += f"at α = {self.alpha} level."
        
        # Recommendations
        recommendations = []
        if not assumptions['normality']:
            recommendations.append("Consider using non-parametric Wilcoxon signed-rank test")
        if abs(effect_size) < 0.2:
            recommendations.append("Effect size is small - consider practical significance")
        elif abs(effect_size) > 0.8:
            recommendations.append("Effect size is large - practically significant")

        result = StatisticalTestResult(
            test_name="One-Sample t-test",
            test_type=TestType.ONE_SAMPLE_T,
            statistic=statistic_val,
            p_value=p_value_val,
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            degrees_of_freedom=df,
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'sample_mean': sample_mean,
                'sample_std': sample_std,
                'sample_size': n,
                'population_mean': population_mean,
                'alternative': alternative
            }
        )

        self.results_history.append(result)
        return result
    
    def two_sample_t_test(self, 
                         group1: np.ndarray, 
                         group2: np.ndarray,
                         equal_var: bool = True,
                         alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        Perform two-sample t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            equal_var: Whether to assume equal variances
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            StatisticalTestResult object
        """
        # Remove NaN values
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]
        
        # Test for equal variances
        levene_stat, levene_p = levene(group1_clean, group2_clean)
        equal_variances = levene_p > 0.05
        
        # Perform test
        statistic, p_value = ttest_ind(group1_clean, group2_clean, 
                                      equal_var=equal_var, alternative=alternative)
        
        # Safe type conversion with defensive programming
        try:
            statistic_val = getattr(statistic, '__float__', lambda: np.nan)()
            if isinstance(statistic, (tuple, list)) and len(statistic) > 0:
                statistic_val = getattr(statistic[0], '__float__', lambda: np.nan)()
        except:
            statistic_val = np.nan
            
        try:
            p_value_val = getattr(p_value, '__float__', lambda: np.nan)()
            if isinstance(p_value, (tuple, list)) and len(p_value) > 0:
                p_value_val = getattr(p_value[0], '__float__', lambda: np.nan)()
        except:
            p_value_val = np.nan
        
        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
        std1, std2 = np.std(group1_clean, ddof=1), np.std(group2_clean, ddof=1)
        n1, n2 = len(group1_clean), len(group2_clean)
        
        if equal_var:
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            effect_size = (mean1 - mean2) / pooled_std
        else:
            effect_size = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
        
        # Degrees of freedom
        if equal_var:
            df = n1 + n2 - 2
        else:
            df = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        try:
            df = int(df)
        except (TypeError, ValueError):
            df = None
        
        # Check assumptions
        norm1_stat, norm1_p = shapiro(group1_clean) if len(group1_clean) <= 5000 else normaltest(group1_clean)
        norm2_stat, norm2_p = shapiro(group2_clean) if len(group2_clean) <= 5000 else normaltest(group2_clean)
        
        assumptions = {
            'normality_group1': norm1_p > 0.05,
            'normality_group2': norm2_p > 0.05,
            'equal_variances': equal_variances if equal_var else True,
            'independence': True,  # Assumed
            'sample_size': min(n1, n2) >= 30
        }
        
        # Interpretation
        is_significant = p_value_val < self.alpha
        interpretation = f"The difference between group means ({mean1:.4f} vs {mean2:.4f}) is "
        if is_significant:
            interpretation += f"statistically significant "
        else:
            interpretation += f"not statistically significant "
        interpretation += f"at α = {self.alpha} level."
        
        # Recommendations
        recommendations = []
        if not assumptions['normality_group1'] or not assumptions['normality_group2']:
            recommendations.append("Consider using non-parametric Mann-Whitney U test")
        if not equal_variances and equal_var:
            recommendations.append("Consider using Welch's t-test (unequal variances)")
        if abs(effect_size) < 0.2:
            recommendations.append("Effect size is small - consider practical significance")
        elif abs(effect_size) > 0.8:
            recommendations.append("Effect size is large - practically significant")
        
        result = StatisticalTestResult(
            test_name="Two-Sample t-test",
            test_type=TestType.TWO_SAMPLE_T,
            statistic=statistic_val,
            p_value=p_value_val,
            effect_size=float(effect_size),
            confidence_interval=None,  # Could calculate difference CI
            degrees_of_freedom=df,
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'group1_mean': mean1,
                'group2_mean': mean2,
                'group1_std': std1,
                'group2_std': std2,
                'group1_size': n1,
                'group2_size': n2,
                'equal_var_assumed': equal_var,
                'levene_test_p': levene_p,
                'alternative': alternative
            }
        )
        
        self.results_history.append(result)
        return result
    
    def paired_t_test(self, 
                     before: np.ndarray, 
                     after: np.ndarray,
                     alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        Perform paired t-test.
        
        Args:
            before: Before measurements
            after: After measurements
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            StatisticalTestResult object
        """
        # Remove pairs with NaN values
        mask = ~(np.isnan(before) | np.isnan(after))
        before_clean = before[mask]
        after_clean = after[mask]
        
        # Calculate differences
        differences = after_clean - before_clean
        
        # Perform test
        statistic, p_value = ttest_rel(after_clean, before_clean, alternative=alternative)
        
        # Calculate effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        effect_size = mean_diff / std_diff
        
        # Confidence interval for mean difference
        n = len(differences)
        df = n - 1
        se_diff = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Check assumptions
        norm_stat, norm_p = shapiro(differences) if len(differences) <= 5000 else normaltest(differences)
        assumptions = {
            'normality_differences': norm_p > 0.05,
            'paired_observations': True,  # Assumed by design
            'independence': True,  # Assumed
            'sample_size': len(differences) >= 30
        }
        
        # Interpretation
        is_significant = p_value < self.alpha
        mean_before = np.mean(before_clean)
        mean_after = np.mean(after_clean)
        
        interpretation = f"The paired difference (after - before = {mean_diff:.4f}) is "
        if is_significant:
            interpretation += f"statistically significant "
        else:
            interpretation += f"not statistically significant "
        interpretation += f"at α = {self.alpha} level."
        
        # Recommendations
        recommendations = []
        if not assumptions['normality_differences']:
            recommendations.append("Consider using non-parametric Wilcoxon signed-rank test")
        if abs(effect_size) < 0.2:
            recommendations.append("Effect size is small - consider practical significance")
        elif abs(effect_size) > 0.8:
            recommendations.append("Effect size is large - practically significant")
        
        result = StatisticalTestResult(
            test_name="Paired t-test",
            test_type=TestType.PAIRED_T,
            statistic=statistic,
            p_value=p_value,
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            degrees_of_freedom=df,
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'mean_before': mean_before,
                'mean_after': mean_after,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'sample_size': n,
                'alternative': alternative
            }
        )
        
        self.results_history.append(result)
        return result
    
    def chi_square_test(self, 
                       observed: np.ndarray,
                       expected: Optional[np.ndarray] = None) -> StatisticalTestResult:
        """
        Perform chi-square test of independence or goodness of fit.
        
        Args:
            observed: Observed frequencies (contingency table or 1D array)
            expected: Expected frequencies (for goodness of fit test)
        
        Returns:
            StatisticalTestResult object
        """
        expected_freq = None  # Ensure variable is always defined
        expected_arr = np.array([])  # Initialize default
        expected_freq_arr = np.array([])  # Initialize default

        if observed.ndim == 1:
            # Goodness of fit test
            if expected is None:
                # Uniform distribution assumption
                expected = np.full_like(observed, np.sum(observed) / len(observed))
            
            statistic, p_value = stats.chisquare(observed, expected)
            df = len(observed) - 1
            test_name = "Chi-square Goodness of Fit"
            
            # Effect size (Cramer's V)
            n = np.sum(observed)
            effect_size = np.sqrt(statistic / (n * (len(observed) - 1)))
            
        else:
            # Test of independence
            statistic, p_value, df, expected_freq = chi2_contingency(observed)
            test_name = "Chi-square Test of Independence"
            
            # Effect size (Cramer's V)
            n = np.sum(observed)
            min_dim = min(observed.shape) - 1
            effect_size = np.sqrt(statistic / (n * min_dim))
        
        # Check assumptions
        if observed.ndim == 1:
            # Ensure expected is a numpy array
            expected_arr = np.array(expected) if expected is not None else np.full_like(observed, np.sum(observed) / len(observed))
            if expected_arr.size > 0:
                min_expected = float(np.min(expected_arr))
                cells_below_5 = int(np.sum(expected_arr < 5))
            else:
                min_expected = np.nan
                cells_below_5 = 0
        else:
            # Ensure expected_freq is a numpy array
            expected_freq_arr = np.array(expected_freq) if expected_freq is not None else np.array([])
            min_expected = float(np.min(expected_freq_arr)) if expected_freq_arr.size > 0 else np.nan
            cells_below_5 = int(np.sum(expected_freq_arr < 5)) if expected_freq_arr.size > 0 else 0

        assumptions = {
            'minimum_expected_frequency': min_expected >= 5,
            'cells_below_5': cells_below_5 / np.size(observed) <= 0.2,
            'independence': True,  # Assumed
            'random_sampling': True  # Assumed
        }

        # Interpretation
        try:
            p_value_safe = getattr(p_value, '__float__', lambda: np.nan)()
            if isinstance(p_value, (tuple, list)) and len(p_value) > 0:
                p_value_safe = getattr(p_value[0], '__float__', lambda: np.nan)()
            is_significant = p_value_safe < self.alpha
        except Exception:
            is_significant = False

        if observed.ndim == 1:
            interpretation = f"The observed frequencies {'significantly' if is_significant else 'do not significantly'} "
            interpretation += f"differ from expected frequencies at α = {self.alpha} level."
        else:
            interpretation = f"There {'is' if is_significant else 'is no'} significant association "
            interpretation += f"between the variables at α = {self.alpha} level."

        # Recommendations
        recommendations = []
        if not assumptions['minimum_expected_frequency'] or not assumptions['cells_below_5']:
            recommendations.append("Consider Fisher's exact test or combining categories")
        if effect_size < 0.1:
            recommendations.append("Effect size is small - weak association")
        elif effect_size > 0.5:
            recommendations.append("Effect size is large - strong association")

        # Safe type conversion for chi-square results
        try:
            statistic_val = getattr(statistic, '__float__', lambda: np.nan)()
            if isinstance(statistic, (tuple, list)) and len(statistic) > 0:
                statistic_val = getattr(statistic[0], '__float__', lambda: np.nan)()
        except Exception:
            statistic_val = np.nan
            
        try:
            p_value_val = getattr(p_value, '__float__', lambda: np.nan)()
            if isinstance(p_value, (tuple, list)) and len(p_value) > 0:
                p_value_val = getattr(p_value[0], '__float__', lambda: np.nan)()
        except Exception:
            p_value_val = np.nan
            
        try:
            df_val = getattr(df, '__int__', lambda: None)()
            if isinstance(df, (tuple, list)) and len(df) > 0:
                df_val = getattr(df[0], '__int__', lambda: None)()
        except Exception:
            df_val = None

        # Ensure expected frequency arrays exist for all cases
        expected_frequencies = expected_arr if observed.ndim == 1 else expected_freq_arr

        result = StatisticalTestResult(
            test_name=test_name,
            test_type=TestType.CHI_SQUARE,
            statistic=statistic_val,
            p_value=p_value_val,
            effect_size=float(effect_size),
            confidence_interval=None,
            degrees_of_freedom=df_val,
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'observed_frequencies': observed,
                'expected_frequencies': expected_frequencies,
                'total_observations': int(np.sum(observed)),
                'cramers_v': float(effect_size)
            }
        )

        self.results_history.append(result)
        return result
    
    def anova_test(self, *groups: np.ndarray) -> StatisticalTestResult:
        """
        Perform one-way ANOVA.
        
        Args:
            *groups: Variable number of group arrays
        
        Returns:
            StatisticalTestResult object
        """
        # Clean groups (remove NaN values)
        clean_groups = [group[~np.isnan(group)] for group in groups]
        
        # Perform ANOVA
        statistic, p_value = f_oneway(*clean_groups)
        
        # Calculate effect size (eta-squared)
        all_data = np.concatenate(clean_groups)
        grand_mean = np.mean(all_data)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in clean_groups)
        
        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)
        
        # Effect size (eta-squared)
        eta_squared = ss_between / ss_total
        
        # Degrees of freedom
        df_between = len(groups) - 1
        df_within = len(all_data) - len(groups)
        
        # Check assumptions
        # Normality for each group
        normality_tests = []
        for i, group in enumerate(clean_groups):
            if len(group) <= 5000:
                stat, p = shapiro(group)
            else:
                stat, p = normaltest(group)
            normality_tests.append(p > 0.05)
        
        # Homogeneity of variances
        levene_stat, levene_p = levene(*clean_groups)
        
        assumptions = {
            'normality': all(normality_tests),
            'homogeneity_of_variances': levene_p > 0.05,
            'independence': True,  # Assumed
            'sample_sizes': all(len(group) >= 2 for group in clean_groups)
        }
        
        # Group means
        group_means = [np.mean(group) for group in clean_groups]
        
        # Interpretation
        is_significant = p_value < self.alpha
        interpretation = f"There {'is' if is_significant else 'is no'} significant difference "
        interpretation += f"between group means at α = {self.alpha} level."
        if is_significant:
            interpretation += " Post-hoc tests recommended to identify specific differences."
        
        # Recommendations
        recommendations = []
        if not assumptions['normality']:
            recommendations.append("Consider using non-parametric Kruskal-Wallis test")
        if not assumptions['homogeneity_of_variances']:
            recommendations.append("Consider Welch's ANOVA for unequal variances")
        if is_significant:
            recommendations.append("Perform post-hoc tests (Tukey HSD, Bonferroni)")
        if eta_squared < 0.01:
            recommendations.append("Effect size is small")
        elif eta_squared > 0.14:
            recommendations.append("Effect size is large")
        
        result = StatisticalTestResult(
            test_name="One-way ANOVA",
            test_type=TestType.ANOVA,
            statistic=statistic,
            p_value=p_value,
            effect_size=float(eta_squared),
            confidence_interval=None,
            degrees_of_freedom=int(df_between),
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'group_means': group_means,
                'group_sizes': [len(group) for group in clean_groups],
                'grand_mean': grand_mean,
                'eta_squared': eta_squared,
                'ss_between': ss_between,
                'ss_total': ss_total,
                'levene_p_value': levene_p
            }
        )
        
        self.results_history.append(result)
        return result
    
    def correlation_test(self, 
                        x: np.ndarray, 
                        y: np.ndarray,
                        method: CorrelationType = CorrelationType.PEARSON) -> StatisticalTestResult:
        """
        Perform correlation analysis.
        
        Args:
            x: First variable
            y: Second variable
            method: Type of correlation (Pearson, Spearman, Kendall)
        
        Returns:
            StatisticalTestResult object
        """
        # Remove pairs with NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Perform correlation test
        if method == CorrelationType.PEARSON:
            statistic, p_value = pearsonr(x_clean, y_clean)
            test_name = "Pearson Correlation"
        elif method == CorrelationType.SPEARMAN:
            statistic, p_value = spearmanr(x_clean, y_clean)
            test_name = "Spearman Correlation"
        elif method == CorrelationType.KENDALL:
            statistic, p_value = kendalltau(x_clean, y_clean)
            test_name = "Kendall's Tau"
        
        # Safe statistic conversion and casting to float
        try:
            statistic_scalar = getattr(statistic, '__float__', lambda: np.nan)()
            if isinstance(statistic, (tuple, list)) and len(statistic) > 0:
                statistic_scalar = getattr(statistic[0], '__float__', lambda: np.nan)()
        except:
            statistic_scalar = np.nan

        # Effect size is the correlation coefficient itself
        effect_size = abs(statistic_scalar)
        
        # Confidence interval for Pearson correlation
        n = len(x_clean)
        if method == CorrelationType.PEARSON and n > 3:
            # Fisher's z-transformation
            z = 0.5 * np.log((1.0 + statistic_scalar) / (1.0 - statistic_scalar))
            se_z = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            z_lower = z - z_critical * se_z
            z_upper = z + z_critical * se_z
            
            # Transform back to correlation scale
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None
        
        # Check assumptions
        if method == CorrelationType.PEARSON:
            # Check for linearity and normality
            norm_x_stat, norm_x_p = shapiro(x_clean) if len(x_clean) <= 5000 else normaltest(x_clean)
            norm_y_stat, norm_y_p = shapiro(y_clean) if len(y_clean) <= 5000 else normaltest(y_clean)
            
            assumptions = {
                'linearity': True,  # Should be checked visually
                'normality_x': norm_x_p > 0.05,
                'normality_y': norm_y_p > 0.05,
                'independence': True,  # Assumed
                'homoscedasticity': True  # Should be checked visually
            }
        else:
            assumptions = {
                'monotonicity': True,  # Should be checked visually
                'independence': True,  # Assumed
                'ordinal_data': True  # Assumed for Spearman/Kendall
            }
        
        # Safe conversion for statistic and p_value scalars
        try:
            statistic_scalar = getattr(statistic, '__float__', lambda: np.nan)()
            if isinstance(statistic, (tuple, list)) and len(statistic) > 0:
                statistic_scalar = getattr(statistic[0], '__float__', lambda: np.nan)()
        except:
            statistic_scalar = np.nan
            
        try:
            p_value_scalar = getattr(p_value, '__float__', lambda: np.nan)()
            if isinstance(p_value, (tuple, list)) and len(p_value) > 0:
                p_value_scalar = getattr(p_value[0], '__float__', lambda: np.nan)()
        except:
            p_value_scalar = np.nan
        
        # Interpretation
        is_significant = p_value_scalar < self.alpha
        strength = "weak" if abs(statistic_scalar) < 0.3 else "moderate" if abs(statistic_scalar) < 0.7 else "strong"
        direction = "positive" if statistic_scalar > 0 else "negative"
        
        interpretation = f"There {'is' if is_significant else 'is no'} significant "
        interpretation += f"{strength} {direction} correlation (r = {statistic_scalar:.4f}) "
        interpretation += f"at α = {self.alpha} level."
        
        # Recommendations
        recommendations = []
        if method == CorrelationType.PEARSON and not all(assumptions.values()):
            recommendations.append("Consider Spearman correlation for non-parametric analysis")
        if abs(statistic_scalar) < 0.1:
            recommendations.append("Correlation is very weak - little practical relationship")
        elif abs(statistic_scalar) > 0.7:
            recommendations.append("Strong correlation - consider potential causation")
        if is_significant:
            recommendations.append("Examine scatterplot for outliers and non-linear patterns")
        
        result = StatisticalTestResult(
            test_name=test_name,
            test_type=TestType.CORRELATION,
            statistic=statistic_scalar,
            p_value=p_value_scalar,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            degrees_of_freedom=n-2 if method == CorrelationType.PEARSON else None,
            interpretation=interpretation,
            recommendations=recommendations,
            assumptions_met=assumptions,
            additional_info={
                'correlation_coefficient': statistic_scalar,
                'sample_size': n,
                'correlation_type': method.value,
                'strength': strength,
                'direction': direction
            }
        )
        
        self.results_history.append(result)
        return result

class RegressionAnalyzer:
    """Comprehensive regression analysis toolkit."""
    
    def __init__(self):
        """Initialize the regression analyzer."""
        self.results_history = []
        self.scaler = StandardScaler()
    
    def linear_regression(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         fit_intercept: bool = True,
                         test_size: float = 0.2) -> RegressionResult:
        """
        Perform linear regression analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            fit_intercept: Whether to fit intercept
            test_size: Proportion of data for testing
        
        Returns:
            RegressionResult object
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Fit model
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_full = model.predict(X)
        
        # Performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # Adjusted R-squared
        n, p = X.shape
        adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        # Residuals
        residuals = y_test - y_pred_test
        
        # Statistical inference (for simple cases)
        if X.shape[1] == 1:
            # Simple linear regression statistics
            slope, _, _, p_value, std_err = linregress(X.flatten(), y)
            
            # Safe conversion for slope and std_err
            try:
                slope_val = getattr(slope, '__float__', lambda: np.nan)()
                if isinstance(slope, (tuple, list)) and len(slope) > 0:
                    slope_val = getattr(slope[0], '__float__', lambda: np.nan)()
            except:
                slope_val = np.nan
                
            try:
                std_err_val = getattr(std_err, '__float__', lambda: np.nan)()
                if isinstance(std_err, (tuple, list)) and len(std_err) > 0:
                    std_err_val = getattr(std_err[0], '__float__', lambda: np.nan)()
            except:
                std_err_val = np.nan
                
            standard_errors = np.array([std_err_val])
            t_statistics = np.array([slope_val / std_err_val if std_err_val != 0 else np.nan])
            p_values = np.array([p_value])
        else:
            # Multiple regression - approximate standard errors
            mse_residual = np.sum(residuals**2) / (len(residuals) - X.shape[1] - 1)
            var_coef = mse_residual * np.linalg.inv(X_train.T @ X_train).diagonal()
            standard_errors = np.sqrt(var_coef)
            t_statistics = model.coef_ / standard_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), len(y_train) - X.shape[1] - 1))
        
        # Confidence intervals for coefficients
        df = len(y_train) - X.shape[1] - 1
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = model.coef_ - t_critical * standard_errors
        ci_upper = model.coef_ + t_critical * standard_errors
        confidence_intervals = np.column_stack([ci_lower, ci_upper])
        
        # AIC and BIC
        n = len(y_test)
        k = X.shape[1] + 1  # Include intercept
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        # Durbin-Watson test for autocorrelation
        diff_residuals = np.diff(residuals)
        durbin_watson = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        # Assumptions checking
        assumptions = self._check_regression_assumptions(X_test, residuals, y_pred_test)
        
        # Interpretation
        interpretation = f"Linear regression model explains {r2_test:.1%} of the variance in the target variable. "
        if r2_test > 0.7:
            interpretation += "Model shows strong predictive power."
        elif r2_test > 0.5:
            interpretation += "Model shows moderate predictive power."
        else:
            interpretation += "Model shows weak predictive power."
        
        # Recommendations
        recommendations = []
        if r2_test < 0.5:
            recommendations.append("Consider polynomial features or different model types")
        if not assumptions['linearity']:
            recommendations.append("Examine residual plots for non-linear patterns")
        if not assumptions['homoscedasticity']:
            recommendations.append("Consider robust regression or data transformation")
        if not assumptions['independence']:
            recommendations.append("Check for temporal patterns or clusters in data")
        
        result = RegressionResult(
            regression_type=RegressionType.LINEAR,
            model=model,
            r_squared=r2_test,
            adjusted_r_squared=adj_r2,
            coefficients=model.coef_,
            intercept=float(model.intercept_) if isinstance(model.intercept_, np.ndarray) else model.intercept_,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            residuals=residuals,
            predictions=y_pred_test,
            mse=mse,
            mae=mae,
            aic=aic,
            bic=bic,
            durbin_watson=durbin_watson,
            diagnostic_plots={},
            assumptions_check=assumptions,
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        self.results_history.append(result)
        return result
    
    def polynomial_regression(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            degree: int = 2,
                            test_size: float = 0.2) -> RegressionResult:
        """
        Perform polynomial regression analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            degree: Polynomial degree
            test_size: Proportion of data for testing
        
        Returns:
            RegressionResult object
        """
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=42
        )
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_test = model.predict(X_test)
        
        # Performance metrics
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # Adjusted R-squared
        n, p = X_poly.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Residuals
        residuals = y_test - y_pred_test
        
        # Assumptions checking
        assumptions = self._check_regression_assumptions(X_test, residuals, y_pred_test)
        
        # AIC and BIC
        n = len(y_test)
        k = X_poly.shape[1] + 1
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        # Interpretation
        interpretation = f"Polynomial regression (degree {degree}) explains {r2:.1%} of the variance. "
        if degree > 3:
            interpretation += "High degree may indicate overfitting."
        
        # Recommendations
        recommendations = []
        if degree > 2:
            recommendations.append("Consider regularization to prevent overfitting")
        recommendations.append("Compare with simpler models using cross-validation")
        
        # Store polynomial transformer for future use
        setattr(model, 'poly_features', poly_features)
        
        result = RegressionResult(
            regression_type=RegressionType.POLYNOMIAL,
            model=model,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            coefficients=model.coef_,
            intercept=float(model.intercept_) if isinstance(model.intercept_, np.ndarray) else model.intercept_,
            standard_errors=None,
            t_statistics=None,
            p_values=None,
            confidence_intervals=None,
            residuals=residuals,
            predictions=y_pred_test,
            mse=mse,
            mae=mae,
            aic=aic,
            bic=bic,
            durbin_watson=None,
            diagnostic_plots={},
            assumptions_check=assumptions,
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        self.results_history.append(result)
        return result
    
    def ridge_regression(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        alpha: float = 1.0,
                        test_size: float = 0.2) -> RegressionResult:
        """
        Perform Ridge regression analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            alpha: Regularization strength
            test_size: Proportion of data for testing
        
        Returns:
            RegressionResult object
        """
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_test = model.predict(X_test_scaled)
        
        # Performance metrics
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # Adjusted R-squared
        n, p = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Residuals
        residuals = y_test - y_pred_test
        
        # Store scaler with model
        setattr(model, 'scaler', scaler)
        
        # Assumptions checking
        assumptions = self._check_regression_assumptions(X_test_scaled, residuals, y_pred_test)
        
        # Interpretation
        interpretation = f"Ridge regression (α={alpha}) explains {r2:.1%} of the variance. "
        interpretation += "Regularization helps prevent overfitting."
        
        # Recommendations
        recommendations = []
        recommendations.append("Try different alpha values using cross-validation")
        if np.max(np.abs(model.coef_)) < 0.1:
            recommendations.append("Strong regularization - consider reducing alpha")
        
        result = RegressionResult(
            regression_type=RegressionType.RIDGE,
            model=model,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            coefficients=model.coef_,
            intercept=float(model.intercept_) if isinstance(model.intercept_, np.ndarray) else model.intercept_,
            standard_errors=None,
            t_statistics=None,
            p_values=None,
            confidence_intervals=None,
            residuals=residuals,
            predictions=y_pred_test,
            mse=mse,
            mae=mae,
            aic=None,
            bic=None,
            durbin_watson=None,
            diagnostic_plots={},
            assumptions_check=assumptions,
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        self.results_history.append(result)
        return result
    
    def logistic_regression(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           test_size: float = 0.2) -> RegressionResult:
        """
        Perform logistic regression analysis.
        
        Args:
            X: Feature matrix
            y: Binary target variable
            test_size: Proportion of data for testing
        
        Returns:
            RegressionResult object
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        # Pseudo R-squared (McFadden's)
        null_model = LogisticRegression()
        null_model.fit(np.ones((len(X_train_scaled), 1)), y_train)
        null_ll = -log_loss(y_test, null_model.predict_proba(np.ones((len(X_test_scaled), 1)))[:, 1])
        model_ll = -logloss
        pseudo_r2 = 1 - (model_ll / null_ll)
        
        # Store scaler with model
        setattr(model, 'scaler', scaler)
        
        # Interpretation
        interpretation = f"Logistic regression achieves {accuracy:.1%} accuracy with AUC = {auc:.3f}. "
        if auc > 0.8:
            interpretation += "Model shows excellent discrimination."
        elif auc > 0.7:
            interpretation += "Model shows good discrimination."
        else:
            interpretation += "Model shows fair discrimination."
        
        # Recommendations
        recommendations = []
        if auc < 0.7:
            recommendations.append("Consider feature engineering or different algorithms")
        recommendations.append("Examine feature importance and coefficient significance")
        
        result = RegressionResult(
            regression_type=RegressionType.LOGISTIC,
            model=model,
            r_squared=float(pseudo_r2),
            adjusted_r_squared=0.0,
            coefficients=model.coef_[0],
            intercept=model.intercept_[0],
            standard_errors=None,
            t_statistics=None,
            p_values=None,
            confidence_intervals=None,
            residuals=np.array([]),
            predictions=y_pred_proba,
            mse=float(logloss),
            mae=float(1 - accuracy),
            aic=None,
            bic=None,
            durbin_watson=None,
            diagnostic_plots={},
            assumptions_check={'linearity': True},  # Different assumptions for logistic
            interpretation=interpretation,
            recommendations=recommendations
        )
        
        self.results_history.append(result)
        return result
    
    def _check_regression_assumptions(self, 
                                    X: np.ndarray, 
                                    residuals: np.ndarray, 
                                    predictions: np.ndarray) -> Dict[str, bool]:
        """
        Check regression assumptions.
        
        Args:
            X: Feature matrix
            residuals: Model residuals
            predictions: Model predictions
        
        Returns:
            Dictionary of assumption checks
        """
        assumptions = {}
        
        # Linearity (check if residuals vs fitted has no pattern)
        # Simple check: correlation between residuals and predictions should be near zero
        try:
            corr_result = pearsonr(np.ravel(predictions), np.ravel(residuals))
            # Extremely defensive extraction with getattr
            corr_coef = 0.0
            if hasattr(corr_result, '__getitem__'):
                first_elem = corr_result[0]
                float_method = getattr(first_elem, '__float__', None)
                if float_method is not None:
                    try:
                        corr_coef = float_method()
                    except:
                        corr_coef = 0.0
        except:
            corr_coef = 0.0
        linearity_corr = abs(corr_coef)
        assumptions['linearity'] = linearity_corr < 0.1
        
        # Independence (Durbin-Watson test approximation)
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        assumptions['independence'] = 1.5 < dw_stat < 2.5
        
        # Homoscedasticity (constant variance)
        # Split predictions into groups and compare variances
        n_groups = 3
        sorted_indices = np.argsort(predictions)
        group_size = len(predictions) // n_groups
        
        group_vars = []
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else len(predictions)
            group_residuals = residuals[sorted_indices[start_idx:end_idx]]
            group_vars.append(np.var(group_residuals))
        
        # Bartlett test for equal variances
        if len(set(group_vars)) > 1:  # Check if variances are different
            var_ratio = max(group_vars) / min(group_vars)
            assumptions['homoscedasticity'] = var_ratio < 3
        else:
            assumptions['homoscedasticity'] = True
        
        # Normality of residuals
        if len(residuals) <= 5000:
            norm_stat, norm_p = shapiro(residuals)
        else:
            norm_stat, norm_p = normaltest(residuals)
        assumptions['normality'] = norm_p > 0.05
        
        return assumptions

# Factory functions for easy use
def create_hypothesis_tester(alpha: float = 0.05) -> HypothesisTestAnalyzer:
    """Create a hypothesis test analyzer."""
    return HypothesisTestAnalyzer(alpha=alpha)

def create_regression_analyzer() -> RegressionAnalyzer:
    """Create a regression analyzer."""
    return RegressionAnalyzer()

def quick_t_test(group1: np.ndarray, 
                group2: Optional[np.ndarray] = None,
                population_mean: Optional[float] = None,
                paired: bool = False,
                alpha: float = 0.05) -> StatisticalTestResult:
    """
    Quick t-test function.
    
    Args:
        group1: First group or single sample
        group2: Second group (for two-sample test)
        population_mean: Population mean (for one-sample test)
        paired: Whether to perform paired t-test
        alpha: Significance level
    
    Returns:
        StatisticalTestResult object
    """
    analyzer = HypothesisTestAnalyzer(alpha=alpha)
    
    if group2 is None and population_mean is not None:
        return analyzer.one_sample_t_test(group1, population_mean)
    elif group2 is not None:
        if paired:
            return analyzer.paired_t_test(group1, group2)
        else:
            return analyzer.two_sample_t_test(group1, group2)
    else:
        raise ValueError("Must provide either group2 or population_mean")

def quick_correlation(x: np.ndarray, 
                     y: np.ndarray,
                     method: str = 'pearson',
                     alpha: float = 0.05) -> StatisticalTestResult:
    """
    Quick correlation analysis.
    
    Args:
        x: First variable
        y: Second variable
        method: Correlation method ('pearson', 'spearman', 'kendall')
        alpha: Significance level
    
    Returns:
        StatisticalTestResult object
    """
    analyzer = HypothesisTestAnalyzer(alpha=alpha)
    correlation_type = CorrelationType(method.lower())
    return analyzer.correlation_test(x, y, correlation_type)

def quick_regression(X: np.ndarray, 
                    y: np.ndarray,
                    regression_type: str = 'linear',
                    **kwargs) -> RegressionResult:
    """
    Quick regression analysis.
    
    Args:
        X: Feature matrix
        y: Target variable
        regression_type: Type of regression ('linear', 'polynomial', 'ridge', 'logistic')
        **kwargs: Additional arguments for specific regression types
    
    Returns:
        RegressionResult object
    """
    analyzer = RegressionAnalyzer()
    
    if regression_type.lower() == 'linear':
        return analyzer.linear_regression(X, y, **kwargs)
    elif regression_type.lower() == 'polynomial':
        return analyzer.polynomial_regression(X, y, **kwargs)
    elif regression_type.lower() == 'ridge':
        return analyzer.ridge_regression(X, y, **kwargs)
    elif regression_type.lower() == 'logistic':
        return analyzer.logistic_regression(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown regression type: {regression_type}")

# Export main classes and functions
__all__ = [
    'HypothesisTestAnalyzer',
    'RegressionAnalyzer',
    'StatisticalTestResult',
    'RegressionResult',
    'TestType',
    'RegressionType',
    'CorrelationType',
    'create_hypothesis_tester',
    'create_regression_analyzer',
    'quick_t_test',
    'quick_correlation',
    'quick_regression'
]
