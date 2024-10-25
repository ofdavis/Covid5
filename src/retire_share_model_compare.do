frame change default
cd "/users/owen/Covid5/"
use data/covid_data, clear 
global covid_line xline(`=tm(2020m4)', lc(black%25) lp(dash))

* keep cpsidp wtfinl retired mo age race educ sex 

merge 1:1 cpsidp mo using "data/generated/retired_share_nn2"
assert _merge==3
drop py_nn
rename py2_nn py_nn
drop _merge

merge 1:1 cpsidp mo using "data/generated/retired_share_ridge", keepusing(py_lr_2lb)
* assert _merge==3
drop if _merge==2 // this is the annoying military guys I threw out -- redo ridge with new data  
rename py_lr_2lb py_ridge 
drop _merge
 
gen age_ = age 
replace age_=80 if age>=80

gen race_ = race 
replace race_ = 4 if race==5 // group asian and other -- diff from Montes Faria

gen educ_ = inrange(educ,4,5)

egen demo = group(age_ race_ sex educ_)
* unique demo 

* make some simplified vars for reporting
gen agegrp = (age>=60) + (age>=65) + (age>=70)
gen natgrp = nativity==2
recode educ (0/1 = 1) (2=2) (3/4=3), gen(educ2)



/*-----------------------------------------------------------------------------
					Collapse by demogs -- for comparison with reg 
-----------------------------------------------------------------------------*/
frame copy default coll, replace 
frame change coll 
gen w = 1 
collapse (mean) retired age_ educ_ race_ sex py_nn py_ridge (sum) w [fw=round(wtfinl)], by(mo demo)

merge 1:1 mo demo using data/generated/retire_model_reg_coll, 
assert _merge==3
drop _merge 

rename retired_p py_reg

*gen year = year(dofm(mo))


* ------------- collapse overall -------------
global pvars py_reg py_ridge py_nn
frame copy coll coll2, replace
frame change coll2 
collapse (mean) retired $pvars age_ year [fw=w], by(mo)
tsset mo 
local varlist "" 
foreach var in retired $pvars {
	tssmooth ma `var'_ma = `var', window(11 1) 
	local varlist = "`varlist' " + "`var'_ma"
}
di "`varlist'"
tsline `varlist' if mo>=`=tm(2010m1)',  lpattern(solid dash dash dash)  ///
	legend(order(1 "Retired share" 2 "Predicted, conventional" 3 "Predicted, ridge" 4 "Predicted, neural net") ///
		size(medsmall) ) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(nn, replace)  xsize(7)
	
graph export output/figs/retired_share_preds.pdf, replace

	
* ------------- collapse overall, region  -------------
global pvars py_reg py_ridge py_nn
frame copy coll coll2, replace
frame change coll2 
collapse (mean) retired $pvars age_ year [fw=w], by(mo)

local pvars_c : subinstr global pvars " " ",", all  // comma-separate the vars 
gen p_retired_min = min(`pvars_c')
gen p_retired_max = max(`pvars_c')

tsset mo 
local varlist "" 
foreach var in retired p_retired_min p_retired_max {
	tssmooth ma `var'_ma = `var', window(11 1) 
	local varlist = "`varlist' " + "`var'_ma"
}

twoway line retired_ma mo if mo>=`=tm(2010m1)' ///
	|| rarea p_retired_min_ma p_retired_max_ma mo if mo>=`=tm(2010m1)', lw(0) fc(black%25) ///,  lpattern(solid dash dash)  ///
	||, legend(order(1 "Retired share" - "" - "" 2 "Predicted," "range" ) size(medsmall)) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(nn, replace) xsize(6)
	
graph export output/figs/retired_share_range.pdf, replace
	

/*-----------------------------------------------------------------------------
					Collapse by variables -- for comparison with reg 
-----------------------------------------------------------------------------*/
* ----------------- program to collapse and plot predicted/actual --------------
cap program drop collplot 
program define collplot 
	syntax varlist, [ by(varlist)]
	frame change default 
	frame copy default collplot, replace 
	frame change collplot 
	do src/labels.do
	*di "`varlist'"
	collapse (mean) retired `varlist' [fw=wtfinl], by(mo `by')
	xtset  `by' mo
	
	* moving average 
	foreach var in retired `varlist' {
		di "tssmooth ma `var'_ma = `var', window(11 1) "
		tssmooth ma `var'_ma = `var', window(11 1) 
	} 
	
	local mlist "" 
	foreach var of local varlist { 
		local mlist = "`mlist'" + "`var'_ma, "
		di"`mlist'"
	}
	local mlist = substr("`mlist'",1,strlen("`mlist'")-2)
	gen p_retired_min_ma = min(`mlist')
	gen p_retired_max_ma = max(`mlist')
	
	* graph 
	qui levelsof `by', local(levels)
	local comb ""
	local num = r(r)
	foreach lev of local levels {
		twoway line retired_ mo if mo>=`=tm(2010m1)' & `by'==`lev' ///
			|| rarea p_retired_min_ma p_retired_max_ma mo if mo>=`=tm(2010m1)' & `by'==`lev', ///
				lw(0) fc(black%25) ///
			||, legend(order(1 "Actual retired share" - "" - "" 2 "Predicted, range") size(medsmall) rows(1)) ///
				xlabel(, format(%tmCY)) xtitle("") $covid_line  ///
				title("${`by'`lev'}", size(medium)) name(`by'`lev', replace) nodraw 
		local comb = "`comb'" + "`by'`lev' "
	}
	
	* graph combine 
	if `num'==2 local setup "cols(1)"
	if `num'==2 local width "xsize(4)"
	if `num'==3 local setup "cols(2) rows(2)"
	if `num'==4 local setup "cols(2) rows(2)"
	if `num'==5 di "too many levels, graph will suck"
	grc1leg2 `comb', `setup' title("${`by'_lab}") imargin(1 1 1 1) `width'
end

foreach var in agegrp sex race_ educ2 married diffphys diffmob { 
	frame change default 
	collplot py_nn py_ridge , by(`var')
	graph export output/figs/retired_share_range_`var'.pdf, replace
}

collplot py_nn py_ridge , by(sex)

