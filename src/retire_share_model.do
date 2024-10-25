* this file replicates the retirement model of Montes et al via Faria e Castro and Jordan-Wood
* see here: https://s3.amazonaws.com/real.stlouisfed.org/wp/2023/2023-010.pdf

frame change default
cd "/users/owen/Covid5/"
use data/covid_data, clear 

*--------------------------Create groups for regs------------------------------* 
gen age_ = age 
replace age_=80 if age>=80

gen race_ = race 
replace race_ = 4 if race==5 // group asian and other -- diff from Montes Faria

gen educ_ = inrange(educ,4,5)

egen demo = group(age_ race_ sex educ_)
* unique demo 


/*-----------------------------------------------------------------------------
								Collapse 
-----------------------------------------------------------------------------*/

* collapse 
gen w = 1 
collapse (mean) retired age_ educ_ race_ sex pia urhat (sum) w [fw=round(wtfinl)], by(mo demo)

gen year = year(dofm(mo))


/*-----------------------------------------------------------------------------
								Regs
-----------------------------------------------------------------------------*/
gen retired_p = . 
gen retired_p2 = . 
levelsof demo, local(demos)
local total = r(r)
*local demos 141
foreach d of local demos { 
	* mo linear 
	cap drop ptmp 
	di "demo `d' out of `total'"
	qui sum age_ if demo==`d' 
	if inrange(r(mean),62,70) { 
		qui reg retired urhat pia mo if demo==`d' & mo<`=tm(2020m1)'
	}
	else { 
		qui reg retired urhat mo if demo==`d' & mo<`=tm(2020m1)'
	}
	qui predict ptmp if demo==`d'
	qui replace retired_p = ptmp if demo==`d'
	
	* 2010+
	cap drop ptmp 
	qui sum age_ if demo==`d' 
	if inrange(r(mean),62,70) { 
		qui reg retired urhat pia c.mo if demo==`d' & mo<`=tm(2020m1)' & mo>`=tm(2010m1)'
	}
	else { 
		qui reg retired urhat c.mo if demo==`d' & mo<`=tm(2020m1)' & mo>`=tm(2010m1)'
	}
	qui predict ptmp if demo==`d'
	qui replace retired_p2 = ptmp if demo==`d'
}
drop ptmp 

* save 
save data/generated/retire_model_reg_coll, replace


/*-----------------------------------------------------------------------------
							Collapse and graph 
-----------------------------------------------------------------------------*/
*frame change default
use data/generated/retire_model_reg_coll, clear 


* ------------- collapse overall -------------
frame copy default coll2, replace
frame change coll2 
collapse (mean) retired retired_p retired_p2 age_ year [fw=w], by(mo)
tsset mo 
tssmooth ma retired_py = retired_p, window(11 1) 
tssmooth ma retired_py2 = retired_p2, window(11 1) 
tssmooth ma retired_ = retired, window(11 1) 
tsline retired_ retired_py retired_py2,  lpattern(solid dash dash)  ///
	legend(order(1 "Actual" - "" - "" 2 "Predicted" 3 "Predicted, v2")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line  name(reg, replace)


* diff 
gen retired_diff = retired-retired_p 
tsline retired_diff, lc(black) lpattern(solid)  ///
	xlabel(, format(%tmCY)) xtitle("") ytitle("") $covid_line 



* ------------- collapse by educ -------------
frame copy default coll2, replace
frame change coll2 
keep if age>=50
collapse (mean) retired retired_p age_ year [fw=w], by(mo educ)
xtset educ mo 
egen retired_py = mean(retired_p), by(year educ)
replace retired_py=. if month(dofm(mo))!=8 & mo<`=tm(2023m8)' & mo>`=tm(2000m1)'
twoway tsline retired retired_py if  educ==0, lcolor(black black) lpattern(solid dash) ///
	|| tsline retired retired_py if  educ==1, lcolor(gray gray) lpattern(solid dash) ///
	||, legend(order(1 "Non-college"  3 "College" - "" - "" 2 "Predicted")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(retire_model_educ, replace)

* diff 
gen retired_diff = retired-retired_p 
tssmooth ma retired_dma = retired_diff, window(6 1)
twoway tsline retired_dma if  educ==0, lcolor(black) lpattern(solid ) ///
	|| tsline retired_dma if  educ==1, lcolor(gray)  lpattern( dash) ///
	||, legend(order(1 "Non-college"  2 "College") pos(6) rows(1)) ///
	xlabel(, format(%tmCY)) xtitle("") ytitle("") $covid_line  ///
	xsize(3) ysize(2.2)  name(educ_diff, replace) title("By education", size(medium))


* ------------- collapse by sex ------------- 
frame copy default coll2, replace
frame change coll2 
keep if age>=50
collapse (mean) retired retired_p age_ year [fw=w], by(mo sex)
xtset sex mo 
egen retired_py = mean(retired_p), by(year sex)
replace retired_py=. if month(dofm(mo))!=8 & mo<`=tm(2023m8)' & mo>`=tm(2000m1)'
twoway tsline retired retired_py if sex==1 , lcolor(black black) lpattern(solid dash) ///
	|| tsline retired retired_py if sex==2 , lcolor(gray gray) lpattern(solid dash) ///
	||, legend(order( 3 "Female" 1 "Male" - "" - "" 2 "Predicted")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(retire_model_sex, replace)

* diff 
gen retired_diff = retired-retired_p 
tssmooth ma retired_dma = retired_diff, window(6 1)
twoway tsline retired_dma if sex==1, lcolor(black) lpattern(solid ) ///
	|| tsline retired_dma if sex==2, lcolor(gray)  lpattern( dash) ///
	||, legend(order(1 "Male"  2 "Female") pos(6) rows(1)) ///
	xlabel(, format(%tmCY)) xtitle("") ytitle("") $covid_line ///
	xsize(3) ysize(2.2) name(sex_diff,replace) title("By sex", size(medium))



* ------------- collapse by age group ------------- 
frame copy default coll2, replace
frame change coll2 
keep if age>=50
local age1 50 
local age2 62 
local age3 65
local age4 70
gen agegrp = (age>=`age1') + (age>=`age2') + (age>=`age3') + (age>=`age4') 
collapse (mean) retired retired_p age_ year [fw=w], by(mo agegrp)
xtset agegrp mo 
egen retired_py = mean(retired_p), by(year agegrp)
replace retired_py=. if month(dofm(mo))!=8 & mo<`=tm(2023m8)' & mo>`=tm(2000m1)'
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway tsline retired retired_py if agegrp==1  , lcolor(black black) lpattern(solid dash) ///
	|| tsline retired retired_py if agegrp==2  , lcolor(gray gray) lpattern(solid dash) ///
	|| tsline retired retired_py if agegrp==3  , lcolor("`r(p3)'" "`r(p3)'") lpattern(solid dash) ///
	|| tsline retired retired_py if agegrp==4  , lcolor("`r(p4)'" "`r(p4)'") lpattern(solid dash) ///
	||, legend(order(7 "Ages 70+" 5 "Ages 65-69" 3 "Ages 62-64" 1 "Ages 50-61" - "" - "" 2 "Predicted")) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(retire_model_agegrp, replace)

* diff 
gen retired_diff = retired-retired_p 
tssmooth ma retired_dma = retired_diff, window(6 1)
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway tsline retired_dma if agegrp==1, lcolor(black) lpattern(solid dash) ///
	|| tsline retired_dma if agegrp==2, lcolor(gray)  lpattern( dash) ///
	|| tsline retired_dma if agegrp==3, lcolor("`r(p3)'") lpattern(shortdash_dot) ///
	|| tsline retired_dma if agegrp==4, lcolor("`r(p4)'") lpattern(longdash_dot) ///
	||, legend(order(1 "50-61" 2 "62-64" 3 "65-69" 4 "70+") pos(6) rows(1)) ///
	xlabel(, format(%tmCY)) xtitle("") ytitle("") $covid_line  ///
	xsize(3) ysize(2.2) name(age_diff,replace) title("By age group", size(medium))



* ------------- collapse by race ------------- 
frame copy default coll2, replace
frame change coll2 
keep if age>=50
collapse (mean) retired retired_p age_ year [fw=w], by(mo race_)
xtset race_ mo 
egen retired_py = mean(retired_p), by(year race_)
replace retired_py=. if month(dofm(mo))!=8 & mo<`=tm(2023m8)' & mo>`=tm(2000m1)'
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway tsline retired retired_py if race_==1 , lcolor(black black) lpattern(solid dash) ///
	|| tsline retired retired_py if race_==2 , lcolor(gray gray) lpattern(solid dash) ///
	|| tsline retired retired_py if race_==3 , lcolor("`r(p3)'" "`r(p3)'") lpattern(solid dash) ///
	||, legend(order(1 "White NH" 3 "Nonwhite NH" 5 "Hispanic" - "" - "" 2 "Predicted" )) ///
	xlabel(, format(%tmCY)) xtitle("") $covid_line name(retire_model_race, replace)

* diff 
gen retired_diff = retired-retired_p 
tssmooth ma retired_dma = retired_diff, window(6 1)
colorpalette cblind, select(1 2 4 3 5 6 7 8 9) nograph
twoway tsline retired_dma if race_==1, lcolor(black) lpattern(solid dash) ///
	|| tsline retired_dma if race_==2, lcolor(gray)  lpattern( dash) ///
	|| tsline retired_dma if race_==3, lcolor("`r(p3)'") lpattern(shortdash_dot) ///
	||, legend(order(1 "White NH" 2 "Nonwhite NH" 3 "Hispanic")  pos(6) rows(1)) ///
	xlabel(, format(%tmCY)) xtitle("") ytitle("") $covid_line  ///
	xsize(3) ysize(2.2) name(race_diff,replace) title("By race", size(medium))


* ------------- combine diff graphs -------------

graph combine age_diff sex_diff educ_diff race_diff 


