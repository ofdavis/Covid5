use data/generated/cps_data.dta, clear 

* retire preds  
merge 1:1 cpsidp mo using data/generated/pred_cps_R
assert _merge==3
drop _merge

* covid deaths 
rename statefip state
merge m:1 state county using data/generated/housing
drop _merge 

* covid deaths 
merge m:1 state county using data/generated/covid
drop _merge 



* -------------------------------- covid/housing collapse --------------------------
frame copy default coll, replace 
frame change coll 
gen pop=1
gen coll = educ>=4
collapse (mean) retired p_retired age (sum) pop (first) dp covidrate deathrate [fw=wtfinl], by(year state county)
gen diff = retired-p_retired

twoway scatter diff covidrate if year>=2024 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff covidrate if year>=2024 [w=pop]

twoway scatter diff dp if year>=2024 [w=pop], mcolor(black%10) mlw(0) ///
	|| lfit    diff dp if year>=2024 [w=pop]

reghdfe diff dp if year==2024  [pw=pop], vce(rob)
reghdfe diff covidrate if year==2024  [pw=pop], vce(rob)



* -------------------------------- housing collapse --------------------------
frame copy default coll, replace 
frame change coll 
gen pop=1
collapse (mean) retired p_retired (sum) pop (first) dp [fw=wtfinl], by(year state county)
gen diff = retired-p_retired

twoway (scatter diff dp if year>=2024 [w=pop], mcolor(black%10) mlw(0)) (lfit diff dp if year>=2024 [w=pop])
reghdfe diff dp if year>=2024  [pw=pop], abs(year)


* -------------------------------- full tsline, quarterly --------------------------
frame change default 
frame copy default coll, replace 
frame change coll 
collapse (mean) retired p_retired (p50) age [fw=wtfinl], by(mo) 
tsset mo
tsline retired p_retired
tsline age


* -------------------------------- full tsline, quarterly --------------------------
frame copy default coll, replace 
frame change coll 
gen tq = yq(year,quarter(dofm(mo)))
format tq %tq
collapse (mean) retired p_retired [fw=wtfinl], by(tq) 
tsset tq
tsline retired p_retired



* -------------------------------- create cohorts --------------------------
frame change default 
gen byear = year-age 

gen cohort = . 
replace cohort = 1 if inrange(byear, 1961, 1970) // 50-59 
replace cohort = 2 if inrange(byear, 1956, 1960) // 60-64
replace cohort = 3 if inrange(byear, 1951, 1955) // 65-69
replace cohort = 4 if inrange(byear, 1946, 1950) // 70-74
replace cohort = 5 if inrange(byear, 1900, 1945) // 75+

label define cohort 1 "50-59" 2 "60-64" 3 "65-69" 4 "70-74" 5 "75+"
label values cohort cohort

* -------------------------------- tsline by variable --------------------------
frame change default 
local var "educ"
frame copy default coll, replace 
frame change coll 
collapse (mean) retired p_retired [fw=wtfinl], by(mo `var')

xtset `var' mo 

tssmooth ma   retired_ma =   retired, window(11 1) 
tssmooth ma p_retired_ma = p_retired, window(11 1) 
gen diff = retired_ma-p_retired_ma 

qui levelsof `var', local(levels)

local text = "" 
local leg = ""

foreach x of local levels {
	local text = "`text' " + "(tsline diff if `var'==`x')"
	local leg = `"   "'
}
di "`text'"
twoway `text'



*------------------------  summarize bar graph --------------------------------
frame change default 
cap frame drop results 
frame create results 

global vlist all sex nativity diffmob race married child_yng educ metro

* gen all = 1
* gen diff = retired-p_retired
foreach var in $vlist { 
	frame copy default coll, replace 
	frame change coll 
	keep if year==2024 
	collapse (mean) diff [fw=wtfinl], by(`var')
	gen class="`var'"
	rename `var' group
	tempfile temp 
	save "`temp'"
	frame results: append using "`temp'"
}
frame change results 

gen order = 0 
replace order=1 if class=="sex" 
replace order=2 if class=="married"
replace order=3 if class=="race" 
replace order=4 if class=="nativity" 
replace order=5 if class=="educ"  
replace order=6 if class=="diffmob" 
replace order=7 if class=="child_yng" 
replace order=8 if class=="metro" 

qui sum diff if class=="all" 
local y = r(mean)
di `y'
graph bar diff if class!="all", over(group) over(class, sort(order)) horiz nofill ysize(6) yline(`y')


foreach var in $vlist { 
	qui tab `var' 
	local n = r(r)
	graph bar diff, over(`var') ///
		xsize(`n') ysize(3) /// 
		ysc(r(-0.01 0.02)) yla(-0.01(0.005)0.02)  ///
		name(`var',replace) title(`var')
}






