clear all 
cd "/users/owen/Covid5"
use data/generated/cps_data.dta, clear 
do src/labels

* retire preds  
merge 1:1 cpsidp mo using data/generated/pred_cps_R
assert _merge==3
drop _merge

* retire preds bootstrap -- recast to save memory 
frame2 boot, replace 
use data/generated/pred_cps_R_boot
recast float p_retired_*, force 
tempfile boot 
save "`boot'"
frame change default 

merge 1:1 cpsidp mo using "`boot'"
assert _merge==3
drop _merge

* for graphing--redo race 
replace race=3 if race==6 // relabel hisp 
label define race2 1 "White" 2 "Black" 3 "Hispanic" 4 "Asian" 5 "Other" , replace
label values race race2 


* ------------------------ tsline with uncertainty bands ---------------------
frame change default
frame copy default coll, replace 
frame change coll 
collapse (mean) retired p_retired* [fw=wtfinl], by(mo) 
tsset mo

* find 90% bounds 
egen p_retired_p05 = rowpctile(p_retired_*), p(5)
egen p_retired_p95 = rowpctile(p_retired_*), p(95)

* unsmoothed with conf int 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway line p_retired mo , lc( "`r(p3)'") ///
	|| rarea p_retired_p05 p_retired_p95 mo, color("`r(p3)'%30") lw(0) ///
	|| line retired mo, lc(black  ) ///
	||, $covid_line xla(,format(%tmCY)) xtitle("") ///
	legend(order(3 "Retired share" 1 "Predicted" 2 "Confidence interval"))  ///
	xsize(6) ysize(3.75)
graph export output/figs/retired_share_main.pdf, replace 

* simple projection / with trends 
cap drop lin_retired*
qui reg retired mo if inrange(mo,`=tm(2010m1)',`=tm(2020m3)')
predict lin_retired 
qui reg retired c.mo##c.mo if inrange(mo,`=tm(2010m1)',`=tm(2020m3)')
predict lin_retired2
qui reg retired c.mo##c.mo##c.mo if inrange(mo,`=tm(2010m1)',`=tm(2020m3)')
predict lin_retired3
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway line lin_retired  mo if mo>=`=tm(2010m1)', lc( "`r(p2)'") lp(dash) ///
	|| line lin_retired2 mo if mo>=`=tm(2010m1)', lc( "`r(p3)'") lp(dash) ///
	///|| line lin_retired3 mo if mo>=`=tm(2010m1)', lc( "`r(p4)'") lp(dash) ///
	|| line 	retired  mo if mo>=`=tm(2010m1)', lc(black  ) ///
	||, $covid_line xla(,format(%tmCY)) xtitle("") ///
	legend(order(3 "Retired share" ///
				 1 "Predicted," "linear" ///
				 2 "Predicted," "quadratic" ))  ///
	xsize(6) ysize(3.75)
graph export output/figs/retired_share_simple_trends.pdf, replace 

twoway line 	retired  mo if mo>=`=tm(2010m1)', lc(black) ///
	|| line lin_retired2 mo if mo>=`=tm(2010m1)', lc(white%0) lp(dash) ///
	||, $covid_line xla(,format(%tmCY)) xtitle("") ///
	legend(order(1 "Retired share")) /// 
	xsize(6) ysize(3.75)
graph export output/figs/retired_share_simple.pdf, replace 

	
/* ----------------------------- other options ---------------------------------
*  graph all boots 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
local text = "" 
forvalues i=0/49 { 
	local text = `"`text' "' + `"(tsline p_retired_`i', lc("`r(p3)'%5"))"'
}
twoway `text' || tsline p_retired retired , lc("`r(p3)'" black) ///
	||, $covid_line xla(,format(%tmCY)) xtitle("") ///
	legend(order(52 "Retired share" 51 "Predicted" 1 "Bootstrap samples")) 
	
* lowess-smoothed  graph 
foreach var in retired p_retired p_retired_p05 p_retired_p95 { 
	qui lowess `var' mo, bw(0.05) nograph gen(lo_`var')
}

colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway line lo_retired mo, lc("`r(p3)'") ///
	|| line lo_p_retired mo , lc(black) ///
	|| rarea lo_p_retired_p05 lo_p_retired_p95 mo, color(gray%50) lw(0) /// 
	legend(order(1 "Retired share" 2 "Predicted" 3 "Confidence interval")) /// 
	 $covid_line xla(,format(%tmCY)) xtitle("") ///
	 ysc(r(0.37 0.44)) yla(0.38(0.02)0.44) xsize(6) //
*/

* -------------------------------- cohort tsline --------------------------
frame change default 

*define cohorts 
cap drop byear cohort
gen byear = year-age 
gen cohort = . 
replace cohort = 0 if inrange(byear, 1971, 1980) // 46-49 -- 46 bc that's 50 in 2024 (byear 1974)
replace cohort = 1 if inrange(byear, 1961, 1970) // 50-59 
replace cohort = 2 if inrange(byear, 1951, 1960) // 60-69
replace cohort = 3 if inrange(byear, 1900, 1950) // 70+

label define cohort 0 "40-49" 1 "50-59" 2 "60-69" 3 "70+", replace 
label values cohort cohort
label variable cohort "Age in 2020"

* collapse 
frame copy default coll, replace 
frame change coll 
collapse (mean) retired p_retired* (mean) age [fw=wtfinl], by(mo cohort) 
xtset cohort mo

* define diffs, diff_ma, diff_ma top and bottom 
gen diff = retired-p_retired
tssmooth ma diff_ma = diff, window(11 1) 
forvalues i=0/49 { 
	gen diff`i' = retired-p_retired_`i'
	tssmooth ma diff_ma`i' = diff`i', window(11 1) 
}
egen diff_ma_p05 = rowpctile(diff_ma*), p(5)
egen diff_ma_p95 = rowpctile(diff_ma*), p(95)

* build graph 
local text "" 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
forvalues c=0/3 { 
	local p = 1+`c'
	local text = `"`text' "' + `"(line diff_ma mo if cohort==`c', lc("`r(p`p')'")) "'
	local text = `"`text' "' + `"(rarea diff_ma_p05 diff_ma_p95 mo if cohort==`c', lw(0) color("`r(p`p')'%20")) "'
}
di `"`text'"'
twoway `text' ||, $covid_line xtitle("") ytitle("") xla(,format(%tmCY)) /// 
	legend(order(- " " 7 "Ages 70+" 5 "Ages 60-69" 3 "Ages 50-59" 1 "Ages 40-49") pos(3)) ///
	xsize(6) ysize(3.75)
graph export output/figs/cohorts.pdf, replace



*------------------------  summarize bar graph --------------------------------
frame change default 
cap frame drop results 
frame create results 

global vlist all sex married educ race foreign metro

cap drop all 
gen all = 1
label define all 1 "Overall" ,  replace
label values all all 
local i 0 
foreach var in $vlist {
	frame copy default coll, replace 
	frame change coll 
	keep if year==2024 
	collapse (mean) retired p_retired* [fw=wtfinl], by(`var')
	decode `var', gen(name)
	gen add = `i'
	local i = `i'+1

	tempfile temp 
	save "`temp'"
	frame results: append using "`temp'"
}
frame change results 

gen diff = retired-p_retired 
forvalues i=0/49 { 
	gen diff_`i' = retired-p_retired_`i'
}
egen diff_p05 = rowpctile(diff_*), p(5)
egen diff_p95 = rowpctile(diff_*), p(95)

cap drop num
gen num = _n*2 + add*2
labmask num, val(name)

*graph
qui levelsof num
local num = r(levels)
di "`num'"
twoway bar diff num, horiz barw(1.5) ///
	|| rcap diff_p05 diff_p95 num, horiz ///
	||, yla("`num'", valuelabel) ysc(reverse) xtitle("") ytitle("") legend(off) xline(0, lc(black)) 
graph export output/figs/diff_demo_bars.pdf, replace 


* -------------------------------- covid/housing  --------------------------
frame change default 

*  housing prices 
rename statefip state
merge m:1 state county using data/generated/housing
drop _merge 

* covid deaths 
merge m:1 state county using data/generated/covid
drop _merge 

frame copy default coll, replace 
frame change coll 
gen pop=1
gen coll = educ>=4
collapse (mean) retired p_retired* age (sum) pop (first) dp covidrate deathrate [fw=wtfinl], by(year state county)
gen diff = retired-p_retired
forvalues i=0/49 { 
	gen diff_`i' = retired-p_retired_`i'
}

* covid deaths 
reg diff covidrate if year==2024 [pw=pop], r
local r2 : display %05.3f e(r2) 
twoway scatter diff covidrate if year==2024 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff covidrate if year==2024 [w=pop] /// 
	||, text(0.16 0.008 "R-squared: `r2'") /// 
	legend(off) xtitle("Cumulative Covid-19 mortality rate") /// 
	ytitle("Excess retirement")
graph export output/figs/covid_scatter.pdf, replace 

* housing 
reg diff dp if year==2024 [pw=pop], r
local r2 : display %05.3f e(r2) 
twoway scatter diff dp if year==2024 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff dp if year==2024 [w=pop] ///
	||, text(0.16 0.7 "R-squared: `r2'") /// 
	legend(off) xtitle("Cumulative county-level housing price gain, 2019-2023") /// 
	ytitle("Excess retirement")
graph export output/figs/housing_scatter.pdf, replace 
	

* run through all bootstrap samples 
frame change coll  
cap frame drop results 
frame create results 
frame results: set obs 51 

foreach var in dp covidrate {
	if "`var'"=="dp" local title "housing price gain"
	if "`var'"=="covidrate" local title "covid mortality"
	frame results: gen n_`var' = _n 
	frame results: gen b_`var' = .  
	frame results: gen ll_`var' = . 
	frame results: gen ul_`var' = .
	
	frame change coll 
	qui reg diff `var' if year==2024 [pw=pop]
	mat results=r(table) 
	qui frame results: replace b_`var'  = results[1,1] if n_`var'==1
	qui frame results: replace ll_`var' = results[5,1] if n_`var'==1
	qui frame results: replace ul_`var' = results[6,1] if n_`var'==1 
	forvalues i=0/49 { 
		local j = `i'+2
		qui reg diff_`i' `var' if year==2024 [pw=pop]
		mat results=r(table) 
		qui frame results: replace b_`var'  = results[1,1] if n_`var'==`j'
		qui frame results: replace ll_`var' = results[5,1] if n_`var'==`j'
		qui frame results: replace ul_`var' = results[6,1] if n_`var'==`j' 
	}
	frame change results 
	sort b_`var' 
	gen nn_`var' = _n 

	twoway scatter b_`var' nn_`var' if  n_`var'!=1 , mc(gray )  ///
		|| scatter b_`var' nn_`var' if  n_`var'==1 , mc(black) ms(S)  ///
		|| rcap ll_`var' ul_`var' nn_`var' if n_`var'!=1, lc(gray )  /// horiz  ///
		|| rcap ll_`var' ul_`var' nn_`var' if n_`var'==1, lc(black)  /// horiz  ///
		||, ytitle("") xtitle("") name(`var',replace) xla("") yline(0,lc(black) lp(dash)) /// ysc(reverse) 
		legend(order(2 "Main estimate" 1 "Bootstrap estimates" ) pos(6) rows(1)) ysize(2) /// 
		title("Coefficient estimates, `title'", size(medsmall))
}
grc1leg2 covidrate dp, rows(2)
graph export output/figs/covid_housing_coeffs_cps.pdf, replace 


* -------------------------------- tsline by var --------------------------
cap program drop collplot 
program define collplot 
	syntax varlist, [ by(varlist)]
	frame change default 
	frame copy default collplot, replace 
	frame change collplot 
	*di "`varlist'"
	collapse (mean) `varlist' p_`varlist'* [fw=wtfinl], by(mo `by')
	xtset  `by' mo
	
	* moving average 
	foreach var in `varlist' p_`varlist' {
		qui tssmooth ma `var'_ma = `var', window(11 1) 
	} 
	forvalues i=0/49 { 
		qui tssmooth ma p_`varlist'_`i'_ma = p_`varlist'_`i', window(11 1)
	}
	
	* get 5th and 95th pctiles of boots 
	egen p_`varlist'_ma_p05 = rowpctile(p_`varlist'_*_ma), p(5)
	egen p_`varlist'_ma_p95 = rowpctile(p_`varlist'_*_ma), p(95)
	
	* graph 
	qui levelsof `by', local(levels)
	local comb ""
	local num = r(r)
	foreach lev of local levels {
		twoway  line `varlist'_ma   mo if mo>=`=tm(2010m1)' & `by'==`lev' ///
			||  line p_`varlist'_ma mo if mo>=`=tm(2010m1)' & `by'==`lev', lp(dash) ///
			||  rarea p_`varlist'_ma_p05 p_`varlist'_ma_p95 mo if mo>=`=tm(2010m1)' & `by'==`lev', ///
				lw(0) color(gray%50) ///
			||, legend(order(1 "Actual retired share" - "" - "" 2 "Predicted") size(medsmall) rows(1)) ///
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
	grc1leg2 `comb', `setup' title("${`by'_lab}") imargin(1 2 1 1) `width' name(`by', replace)
	frame change default 
	frame drop collplot 
end

frame change default
collplot retired, by(educ)


foreach var in sex married foreign race educ metro { 
	collplot retired, by(`var')
} 
/*
sex0  sex1  marr0 marr1 
educ1 educ2 educ3 educ4 educ5 
race1 race2 race3 race4 race5
for0  for1  met0  met1

*/


/* -------------------------------- full tsline, quarterly --------------------------
frame copy default coll, replace 
frame change coll 
gen tq = yq(year,quarter(dofm(mo)))
format tq %tq
collapse (mean) retired p_retired [fw=wtfinl], by(tq) 
tsset tq
tsline retired p_retired
