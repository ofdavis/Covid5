clear all 
cd "/users/owen/Covid5"
use data/generated/asec_data.dta, clear 
do src/labels


* bring in housing data 
rename statefip state
merge m:1 state county using data/generated/housing
assert _merge==3 if year==2024 // some earlier counties don't merge, nbd 
drop _merge 

* covid deaths 
merge m:1 state county using data/generated/covid
drop _merge 

* bring in main predictions 
merge 1:1 asecidp mo using data/generated/pred_asec_R
assert _merge==3
drop _merge 

* bring in bootstrap preds 
merge 1:1 asecidp mo using data/generated/pred_asec_R_boot
assert _merge==3
drop _merge 


* --------------------------- check CPS: overall ------------------------------* 
frame change default
frame copy default coll, replace 
frame change coll 
collapse (mean) retired p_retired*  [fw=asecwt], by(year) 
tsset year


* find 90% bounds 
egen p_retired_p05 = rowpctile(p_retired_*), p(5)
egen p_retired_p95 = rowpctile(p_retired_*), p(95)


* unsmoothed with conf int 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway line p_retired year , lc( "`r(p3)'") ///
	|| rarea p_retired_p05 p_retired_p95 year, color("`r(p3)'%30") lw(0) ///
	|| line retired year, lc(black  ) ///
	||, $covid_line  xtitle("") ///
	legend(order(3 "Retired share" 1 "Predicted" 2 "Confidence interval"))  ///
	xsize(6) ysize(3.75)
graph export output/figs/retired_share_main_asec.pdf, replace 


* --------------------------- check CPS: cohort ------------------------------* 
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
collapse (mean) retired p_retired* (mean) age [fw=asecwt], by(year cohort) 
xtset cohort year

* define diffs, diff_ma, diff_ma top and bottom 
gen diff = retired-p_retired
forvalues i=0/49 { 
	gen diff`i' = retired-p_retired_`i'
}
egen diff_p05 = rowpctile(diff*), p(5)
egen diff_p95 = rowpctile(diff*), p(95)

* cohort diff graph 
local text "" 
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
forvalues c=0/3 { 
	local p = 1+`c'
	local text = `"`text' "' + `"(line diff year if cohort==`c', lc("`r(p`p')'")) "'
	local text = `"`text' "' + `"(rarea diff_p05 diff_p95 year if cohort==`c', lw(0) color("`r(p`p')'%20")) "'
}
di `"`text'"'
twoway `text' ||, $covid_line xtitle("") ytitle("") xla(,format(%tmCY)) /// 
	legend(order(- " " 7 "Ages 70+" 5 "Ages 60-69" 3 "Ages 50-59" 1 "Ages 40-49") pos(3)) ///
	xsize(6) ysize(3.75)
graph export output/figs/cohorts_asec.pdf, replace


* --------------------------- check CPS: bar graph demogs ------------------------------* 
*								  adds incrd and housing
frame change default 
cap frame drop results 
frame create results 

label define incrd 0 "No rent/dividends" 1 "Rent/dividend income", replace
label values incrd incrd 

label define own 0 "Rents" 1 "Owns home", replace
label values own own

global vlist all sex married educ race foreign metro incrd own 

cap drop all 
gen all = 1
label define all 1 "Overall" ,  replace
label values all all 
local i 0 
foreach var in $vlist {
	frame copy default coll, replace 
	frame change coll 
	keep if year==2024 
	collapse (mean) retired p_retired* [fw=asecwt], by(`var')
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

* graph 
qui levelsof num
local num = r(levels)
di "`num'"
twoway bar diff num, horiz barw(1.5) ///
	|| rcap diff_p05 diff_p95 num, horiz ///
	||, yla("`num'", valuelabel) ysc(reverse) xtitle("") ytitle("") legend(off) xline(0, lc(black)) 
graph export output/figs/diff_demo_bars_asec.pdf, replace 


* -------------------------------- covid/housing collapse --------------------------
frame change default 

frame copy default coll, replace 
frame change coll 
gen pop=1
gen coll = educ>=4
collapse (mean) retired p_retired* age (sum) pop (first) dp covidrate deathrate [fw=asecwt], by(year state county own)
gen diff = retired-p_retired
forvalues i=0/49 { 
	gen diff_`i' = retired-p_retired_`i'
}

* create insamp var for graphs to exclude outliers from figure (for readability)
* does not affect regression line 
qui sum diff [fw=pop], d 
cap drop insamp
gen insamp=inrange(diff,r(p1),r(p99))

* covid deaths 
reg diff covidrate if year==2024  [pw=pop]
local r2 : display %05.3f e(r2) 
twoway scatter diff covidrate if year==2024 & insamp==1 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff covidrate if year==2024 & insamp==1 [w=pop] /// 
	||, text(0.1 0.008 "R-squared: `r2'") /// 
	legend(off) xtitle("Cumulative Covid-19 mortality rate") /// 
	ytitle("Excess retirement")
graph export output/figs/covid_scatter_asec.pdf, replace 

* housing 
reg diff dp if year==2024 & own==1 [pw=pop]
local r2 : display %05.3f e(r2) 
twoway scatter diff dp if year==2024 & insamp==1 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff dp if year==2024 & insamp==1 [w=pop] ///
	||, text(0.05 0.7 "R-squared: `r2'") /// 
	legend(off) xtitle("Cumulative county-level housing price gain, 2019-2023") /// 
	ytitle("Excess retirement")
graph export output/figs/housing_scatter_asec.pdf, replace 
	

* run through all bootstrap samples 
frame change coll  
cap frame drop results 
frame create results 
frame results: set obs 51 

foreach var in dp covidrate {
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
		legend(order(2 "Main estimate" 1 "Bootstrap estimates" ) pos(6) rows(1)) ysize(2)
}
grc1leg2 covidrate dp, rows(2)
graph export output/figs/covid_housing_coeffs_asec.pdf, replace 






