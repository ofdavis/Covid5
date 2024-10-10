frame change default
cd "/users/owen/Covid5/"
use data/covid_data, clear 

keep cpsidp wtfinl retired mo age race educ sex 

merge 1:1 cpsidp mo using "data/generated/retired_share_nn"


gen age_ = age 
replace age_=80 if age>=80

gen race_ = race 
replace race_ = 4 if race==5 // group asian and other // diff from Montes Faria

gen educ_ = inrange(educ,4,5)

egen demo = group(age_ race_ sex educ_)
* unique demo 


/*-----------------------------------------------------------------------------
							Collapse and bring in urate 
-----------------------------------------------------------------------------*/

* collapse 
gen w = 1 
collapse (mean) retired age_ educ_ race_ sex py2 (sum) w [fw=round(wtfinl)], by(mo demo)

gen year = year(dofm(mo))


* ------------- collapse overall -------------
frame copy default coll2, replace
frame change coll2 
collapse (mean) retired py2 age_ year [fw=w], by(mo)
tsset mo 
tssmooth ma retired_p = py2, window(11 1) 
tssmooth ma retired_ = retired, window(11 1) 
tsline retired_ retired_p,  lpattern(solid dash dash)  ///
	legend(order(1 "Actual" - "" - "" 2 "Predicted" 3 "Predicted, v2")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(nn, replace)



* ------------- collapse by educ -------------
frame copy default coll2, replace
frame change coll2 
collapse (mean) retired py2 age_ year [fw=w], by(mo educ)
xtset educ mo 
tssmooth ma retired_p = py2, window(11 1) 
tssmooth ma retired_ = retired, window(11 1) 
twoway tsline retired_ retired_p if  educ==0, lcolor(black black) lpattern(solid dash) ///
	|| tsline retired_ retired_p if  educ==1, lcolor(gray gray) lpattern(solid dash) ///
	||, legend(order(1 "Non-college"  3 "College" - "" - "" 2 "Predicted")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(retire_model_educ, replace)
	
