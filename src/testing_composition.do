* a little bit of testing motivated by the diff between Montes et al and the ML estimates 

frame change default
cd "/users/owen/Covid5/"
use data/generated/cps_data, clear 

* for graphing--redo race 
replace race=3 if race==6 // relabel hisp 
label define race2 1 "White" 2 "Black" 3 "Hispanic" 4 "Asian" 5 "Other" , replace
label values race race2 

* -------------------------- collapse all  -------------------- * 
frame change default 
frame copy default coll, replace 

* create dummies for categ vars
tab race, gen(race)
tab educ, gen(educ)

collapse (mean) vet married foreign metro race1 race2 race3 race4 race5 ///
	educ1 educ2 educ3 educ4 educ5 [fw=wtfinl], by(mo)
tsset mo 

foreach var in vet married foreign metro race1 race2 race3 race4 race5 ///
				educ1 educ2 educ3 educ4 educ5 { 
	tsline `var', name(`var',replace)
}


* -------------------------- collapse noncollege  -------------------- * 
frame change default 
frame copy default coll, replace 

keep if educ<3

* create dummies for categ vars
tab race, gen(race)

collapse (mean) vet married foreign metro race1 race2 race3 race4 race5 ///
	 [fw=wtfinl], by(mo)
tsset mo 

foreach var in vet married foreign metro race1 race2 race3 race4 race5 ///
				educ1 educ2 educ3 educ4 educ5 { 
	tsline `var', name(`var',replace)
}
