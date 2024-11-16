import sys
import pandas as pd
from scipy import stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    searchdata = pd.read_json(searchdata_file, orient='records', lines=True)
    
    # Users with an odd-numbered uid were shown a new-and-improved search box. Others were shown the original design.
    
    # Q1. Did more users use the search feature? (More precisely: did a different fraction of users have search count > 0?)
    # Use Mann Whitney U Test
    # separate the users with new search and users with original search
    new_search = searchdata[searchdata['uid']%2==1]
    old_search = searchdata[searchdata['uid']%2==0]
    more_users = stats.mannwhitneyu(new_search['login_count'], old_search['login_count'])
    # pvalue = 0.215 > 0.05 so users use the new search as frequently as the old search
    
    # Q2. Did users search more often? (More precisely: is the number of searches per user different?)
    """Contingency table
                            Searched at least once      Never searched
    New Search (odd uid)    
    Old Search (even uid)
    """     
    new_search_at_least_once = new_search[new_search['search_count']>0]['uid'].count()
    new_search_never = new_search[new_search['search_count']==0]['uid'].count()
    old_search_at_least_once = old_search[old_search['search_count']>0]['uid'].count()
    old_search_never = old_search[old_search['search_count']==0]['uid'].count()
    contingency = [[new_search_at_least_once, new_search_never],
                   [old_search_at_least_once, old_search_never]]
    chi2, more_searches_pvalue, dof, expected = stats.chi2_contingency(contingency)
    # p = 0.168 > 0.05 so there is probably no difference
    
    # Q3. Repeat the above analysis looking only at instructors.
    # Mann-Whiteney U Test
    more_instr = stats.mannwhitneyu(new_search[new_search['is_instructor']]['login_count'], old_search[old_search['is_instructor']]['login_count'])
    # p = 0.899 > 0.05 so no difference
    
    # Chi-square test
    new_search_instr = new_search[new_search['is_instructor']==True]
    old_search_instr = old_search[old_search['is_instructor']==True]
    
    new_search_instr_at_least_once = new_search_instr[new_search_instr['search_count']>0]['uid'].count()
    new_search_instr_never = new_search_instr[new_search_instr['search_count']==0]['uid'].count()
    
    old_search_instr_at_least_once = old_search_instr[old_search_instr['search_count']>0]['uid'].count()
    old_search_instr_never = old_search_instr[old_search_instr['search_count']==0]['uid'].count()
    
    contingency_instr = [[new_search_instr_at_least_once, new_search_instr_never],
                         [old_search_instr_at_least_once, old_search_instr_never]]
    chi2_instr, more_searches_instr_pvalue, dof_instr, expected_instr = stats.chi2_contingency(contingency_instr)
    # p = 0.052 > 0.05 so no difference
    
    
    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users.pvalue,
        more_searches_p=more_searches_pvalue,
        more_instr_p=more_instr.pvalue,
        more_instr_searches_p=more_searches_instr_pvalue,
    ))


if __name__ == '__main__':
    main()
