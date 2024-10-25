clear all 
global path "/Users/owen/Covid5/"
cd "${path}/data/HRS/raw/sects"

* program to define year, wave letter 
cap program drop waveyr 
program waveyr , rclass
	args yr 
	scalar year = 2000+`yr'
	if `yr'==16 local w "p"
	if `yr'==18 local w "q"
	if `yr'==20 local w "r"
	if `yr'==22 local w "s"
	local rw = `yr'/2 + 5
	return scalar yr=`yr' 
	return scalar year=year 
	return scalar rw=`rw'
	return local  w "`w'"
end 

* program that creates a variable named "name" meeting condition at any point within byvar (ie, hhidpn)
cap program drop ever
program define ever
	syntax, name(string) condition(string) byvar(string) maxmin(string) [svar(string)]
	cap drop `name'_ 
	cap drop `name'
	gen `name'_=`condition'
	bys `byvar' (`svar'): egen `name'=`maxmin'(`name'_)
	drop `name'_
end


* ------------------------------ bring in tracker ------------------------------
use tracker.dta, clear 

* keep tracker variables desired 
local keeplist = "birthmo birthyr degree gender race hispanic usborn knowndeceasedmo knowndeceasedyr hhid pn "
foreach w in p q r s {
	local keeplist = "`keeplist' " + "`w'age `w'marst `w'couple `w'nurshm `w'iwwave `w'iwmonth `w'iwyear `w'subhh `w'ppn `w'insamp `w'wgtr `w'alive `w'subhh `w'subhhiw "
}
keep `keeplist'

* intw dates 
foreach w in p q r s {
	gen `w'iwdate = ym(`w'iwyear, `w'iwmonth)
	form `w'iwdate %tm
}

* drop those not appearing in 2016+
keep if (piwwave==1 | qiwwave==1 | riwwave==1 | siwwave==1) & !inrange(palive,5,6) 

* rename to fit rand pattern 
rename pn _pn 
rename race _race
rename p* t13*
rename q* t14* 
rename r* t15*
rename s* t16*

rename _pn pn 
rename _race race

foreach var in birthmo birthyr degree gender hispanic knowndeceasedmo knowndeceasedyr race usborn {
	rename `var' t_`var'
}

* -------------------------------- bring in rand -------------------------------
frame2 rand, replace 
use rand.dta, clear 

local keeplist "hhid pn hhidpn ragender rarace rabplace raeduc r*mstat r*cendiv r*urbrur h*amort h*child h*hhres h*achck h*atotf h*icap h*ahous h*atoth" 
foreach p in r s {
	local keeplist = "`keeplist' " + "`p'*agem_m `p'*jhours `p'*slfemp `p'*fsize `p'*union `p'*wgihr `p'*wgiwk `p'*jcoccc `p'*jcindc `p'*jloccc `p'*jlindc `p'*jcten `p'*jcpen `p'*lbrf `p'*samemp `p'*jlastm `p'*jlasty `p'*sayret `p'*rplnya `p'*retemp `p'*shlt `p'*hltc3 `p'*hibp `p'*diab `p'*cancr `p'*lung `p'*heart"
} 
keep `keeplist'

rename r*hltc3 r*hltc
rename s*hltc3 s*hltc

tempfile rand 
save "`rand'"
frame change default 
merge 1:1 hhid pn using "`rand'"

foreach w in 13 14 15 {	// ensure that those not matching to rand have zero weights in 2016-2020 
	qui sum t`w'wgtr if _merge==1 
	assert r(mean)==0
}
drop if _merge!=3 
drop _merge 

frame drop rand 

/* -----------------------------------------------------------------------------
								CORE FILES
----------------------------------------------------------------------------- */

* ------------------------------ preload: child member  ------------------------
forvalues yr=16(2)22 {
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'PR_MC, clear 
	
	* rename subhh 
	rename `w'subhh t`rw'subhh

	* keep only actual children 
	keep if inlist(`w'x061_mc, 3, 4, 5, 6, 7, 8, 9, 30, 31, 90, 91) & `w'x056_mc==1

	gen age = `year'-`w'x067_mc
	replace age = 0 if age<0 
	replace age = 85 if age>85 & age<. 

	*-------bring in other data for age impute ------
	frame copy default ageimp, replace 
	frame change ageimp
	*keep if `w'wgtr>0 & `w'wgtr<.
	keep hhid t`rw'subhh t_birthyr t`rw'marst t`rw'iwyear t_degree 
	gen  agep = t`rw'iwyear - t_birthyr
	gen nparents = 1
	collapse (sum) nparents (max) max_degree=t_degree max_age=age (min) min_age = age ///
			 (first) marst=t`rw'marst, by(hhid t`rw'subhh)
	tempfile ageimp 
	save "`ageimp'"
	frame change core 
	merge m:1 hhid t`rw'subhh using "`ageimp'"
	drop if _merge==2 

	* relations 
	gen reltype = . 
	replace reltype = 1 if inlist(`w'x061_mc,3,6)
	replace reltype = 2 if inlist(`w'x061_mc,4,5,7,8)
	replace reltype = 3 if inlist(`w'x061_mc,9)
	replace reltype = 4 if inlist(`w'x061_mc,30,31,90,91)
	
	* predict 
	gen ch18 = age<=18 if age!=.
	qui logit ch18 i.nparents i.marst i.max_degree#(c.max_age c.min_age) i.reltype##(c.max_age##c.min_age )

	predict pr_ch18
	gen p_ch18 = pr_ch18>=0.5

	tab ch18 p_ch18

	replace ch18 = p_ch18 if ch18==.
	gen adult = ch18==0

	collapse (sum) h`rw'numch=ch18 h`rw'numad=adult, by(hhid t`rw'subhh)

	tempfile temp 
	save "`temp'"

	frame change default 
	merge m:1 hhid t`rw'subhh using "`temp'", nogen 
	replace h`rw'numch = 0 if h`rw'numch==. & t`rw'iwwave==1 // replace with 0 if not in 
	replace h`rw'numad = 0 if h`rw'numad==. & t`rw'iwwave==1 
}

* ------------------------------ A: Coverscreen, H  ------------------------
* HH: xa020 xa023 xa025 xa036 xa037 xa071 xa072 xa076m
forvalues yr=16(2)22 { 
	*local yr 16
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'A_H, clear 
	keep hhid `w'subhh `w'a020 `w'a023 `w'a024 `w'a025 `w'a036 `w'a037 `w'a045 ///`w'a071 `w'a072 `w'a076m
		 `w'a026 `w'a027 `w'a034 `w'a035 
	
	* whether same spouse/partner (a020)
	gen h`rw'spp_same = inlist(`w'a020,1,3,4) if `w'a020<.
	
	* prev wave spouse/partner still alive (a023)
	gen h`rw'spp_alive = inlist(`w'a023,1,3) if `w'a023<.
	
	* mo/year stopped living together/died 
	gen h`rw'date_breakup = ym(`w'a025,`w'a024) if inrange(`w'a024,1,12) & inrange(`w'a025,1900,2024) 
	replace h`rw'date_breakup = .f if inlist(`w'a024,-8,97,98,99,998,999) // incl those who still/never lived together
	replace h`rw'date_breakup = .f if inlist(`w'a024,-8,9998,9999)
	
	* mo/year new sp/p 
	gen h`rw'date_movein = ym(`w'a037,`w'a036) if inrange(`w'a036,1,12) & inrange(`w'a037,1900,2024) 
	replace h`rw'date_movein = .f if inlist(`w'a024,-8,97,98,99) // incl those who still live or never lived together
	replace h`rw'date_movein = .f if inlist(`w'a024,-8,9998,9999)
	
	* whether new partner (a045)
	gen h`rw'newspp = `w'a045==1 if `w'a045<.
	
	* whether married (a026)
	gen h`rw'married = `w'a026==1 if `w'a026<.
	
	* whether partnered (a027)
	gen h`rw'partner = `w'a027==1 if `w'a027<.
	
	* whether partnered (a034)
	gen h`rw'marrsep = `w'a034 if inrange(`w'a034<.,1,2)
	label define marrsep 1 "married" 2 "separated"
	label values h`rw'marrsep marrsep
	
	* whether partnered (if sep) (a035)
	gen h`rw'partner2 = `w'a035==1 if `w'a035<.
	
	keep hhid `w'subhh h`rw'spp_same h`rw'spp_alive h`rw'date_breakup h`rw'date_movein h`rw'newspp ///
		 h`rw'married h`rw'partner h`rw'marrsep h`rw'partner2 
	
	rename `w'subhh t`rw'subhh
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid t`rw'subhh using "`temp'"
	drop if _merge ==2 
	drop _merge 
	format h`rw'date_breakup %tm
	format h`rw'date_movein %tm
}


/* ------------------------------ B: Demogs, R  ------------------------*/
forvalues yr=16(2)22 { 
	*local yr 16
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'B_R, clear 
	keep hhid pn `w'b055 `w'b056 `w'b057 `w'b058 `w'b059 `w'b060 `w'b063
	
	* new marriage (b055)
	gen r`rw'newmarriage = `w'b055==1 if `w'b055<.

	* new marriage month/year 
	gen r`rw'date_newmarr = ym(`w'b057, `w'b056) if inrange(`w'b057,1900,2024) & inrange(`w'b056,1,12)
	replace r`rw'date_newmarr = .f if (inlist(`w'b057,-8,9998,9999) | inlist(`w'b056,-8,98,99)) ///
		& r`rw'newmarriage==1
		
	* newly divorced/widowed (b058)
	gen r`rw'newdivorce = `w'b058==1 if `w'b058<.
	gen r`rw'newwidow   = `w'b058==2 if `w'b058<.

	* new marriage month/year 
	gen r`rw'date_divwid = ym(`w'b060, `w'b059) if inrange(`w'b060,1900,2024) & inrange(`w'b059,1,12)
	replace r`rw'date_divwid = .f if (inlist(`w'b060,-8,9998,9999) | inlist(`w'b059,-8,98,99)) ///
		& (r`rw'newdivorce==1 | r`rw'newwidow==1)
		
	* marital assigned
	label define mstat_ass 1 "married" 2 "annulled" 3 "separated" 4 "divorced" 5 "widowed" ///
		6 "never" 7 "other" 8 "dk" 9 "rf", replace
	gen r`rw'mstat_ass = `w'b063 
	label values r`rw'mstat_ass mstat_ass
	
	keep hhid pn r`rw'newmarriage r`rw'date_newmarr r`rw'newdivorce r`rw'newwidow r`rw'date_divwid r`rw'mstat_ass
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid pn using "`temp'"
	drop if _merge ==2 
	drop _merge 
	format r`rw'date_newmarr r`rw'date_divwid %tm
}


* create r16mstat to match rand mstat (approximately)
gen r16mstat=. 
replace r16mstat=1 if r16mstat_ass==1 
replace r16mstat=4 if r16mstat_ass==3 
replace r16mstat=5 if r16mstat_ass==4 
replace r16mstat=7 if r16mstat_ass==5 
replace r16mstat=8 if r16mstat_ass==6 

label values r16mstat MSTATUS // use Rand label 

*------------------------ aligning marital status ---------------------------- *
* assign partnered from partner var 
replace r16mstat=3 if h16partner==1 | h16partner2==1

* for missings, take r15 if possible 
replace r16mstat=r15mstat if r16mstat==. & t16iwwave==1 & t15iwwave==1 ///
	& h16spp_same!=0 & h16spp_alive!=0 & h16date_breakup==. & h16date_movein==. & h16newspp!=1
	
* remaining missings: logical edits 
replace r16mstat=5 if inrange(r15mstat,1,3) & h16date_breakup<.
replace r16mstat=1 if r16mstat==. & h16newspp==1

* replace h16mstat=partnered if h15mstat=partnered and no other events 
replace r16mstat=3 if r15mstat==3 & !inrange(r16mstat,1,2) ///
	& h16spp_same!=0 & h16spp_alive!=0 & h16date_breakup==. & r16date_divwid==.

* --------------------- assigning dates for marital changes --------------------*
* Note that these situations do NOT have to cover all cases. Goal is to identify 
* most likely changes 

* RULE 1: NEW MARRIAGES/PARTNERSHIPS 
*		  current status married/partnered AND 
* 	      prior status not married/partnered AND 
*		  date_movein or date_newmarr <. AND 
*		  date_breakup==.
* RULE 2: REMARRIAGES/REPARTNERSHIPS
*		  current status married/partnered AND 
* 	      prior status also married/partnered AND 
*	      prior status != current status AND 
*		  date_movein or date_newmarriage <. AND 
*		  date_breakup<. 
* a) first try date_movein if between wave dates 
* b) then if still missing, try date_newmarr if between wave dates 
* c) then if both out of bounds (or oob & missing), use midpoint of waves 
forvalues w1=14/16 {
	*local w1 14 
	local w0 = `w1'-1
	cap drop r`w1'date_mchg1 
	cap drop r`w1'mchg1
	gen r`w1'date_mchg1=. 
	gen r`w1'mchg1 = 0 if t`w1'iwwave==1 
	local cond1 "inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,4,8) & (h`w1'date_movein<. | r`w1'date_newmarr<.) & h`w1'date_breakup==. & t`w0'iwdate<. & t`w1'iwdate<."
	local cond2 "inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,1,3) & r`w1'mstat!=r`w0'mstat & (h`w1'date_movein<. | r`w1'date_newmarr<.) & h`w1'date_breakup<. & t`w0'iwdate<. & t`w1'iwdate<."
	local cond "((`cond1') | (`cond2'))"

	* a) first try date_movein if between wave dates 
	replace r`w1'date_mchg1 = h`w1'date_movein ///
		if `cond' & inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1 ///
		if `cond' & inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)

	* b) then if still missing, try date_newmarr if between wave dates 
	replace r`w1'date_mchg1 = r`w1'date_newmarr ///
		if r`w1'date_mchg1==. & `cond' & inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1 ///
		if r`w1'mchg1==0 & `cond' & inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate)

	* c) then if both out of bounds (or oob & missing), use midpoint of waves 
	replace r`w1'date_mchg1 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg1==. & `cond' & !inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1  ///
		if r`w1'mchg1==0  & `cond' & !inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
									   
	* d) all the above don't work but there's .f (for both conditions, rewritten to specify .f)
	replace r`w1'date_mchg1 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg1==. & inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,4,8) ///
		 & (h`w1'date_movein==.f | r`w1'date_newmarr==.f) & h`w1'date_breakup==. & t`w0'iwdate<. & t`w1'iwdate<.
	replace r`w1'mchg1 = 1  ///
		if r`w1'mchg1==0 & inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,4,8) ///
		 & (h`w1'date_movein==.f | r`w1'date_newmarr==.f) & h`w1'date_breakup==. & t`w0'iwdate<. & t`w1'iwdate<.
		
	replace r`w1'date_mchg1 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg1==. & inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,1,3) & r`w1'mstat!=r`w0'mstat ///
		 & (h`w1'date_movein==.f | r`w1'date_newmarr==.f) & h`w1'date_breakup<. & t`w0'iwdate<. & t`w1'iwdate<.
	replace r`w1'mchg1 = 1  ///
		if r`w1'mchg1==0 & inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,1,3) & r`w1'mstat!=r`w0'mstat ///
		 & (h`w1'date_movein==.f | r`w1'date_newmarr==.f) & h`w1'date_breakup<. & t`w0'iwdate<. & t`w1'iwdate<.
	
	format r`w1'date_mchg1 %tm
}
* redo the above for skip-wave 
forvalues w1=15/16 {
	*local w1 15 
	local w_ = `w1'-1
	local w0 = `w1'-2
	local cond1 "inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,4,8) & (h`w1'date_movein<. | r`w1'date_newmarr<.) & h`w1'date_breakup==. & t`w0'iwdate<. & t`w1'iwdate<. & t`w_'iwdate==."
	local cond2 "inrange(r`w1'mstat,1,3) & inrange(r`w0'mstat,1,3) & r`w1'mstat!=r`w0'mstat & (h`w1'date_movein<. | r`w1'date_newmarr<.) & h`w1'date_breakup<. & t`w0'iwdate<. & t`w1'iwdate<."
	local cond  "((`cond1') | (`cond2'))"

	* a) first try date_movein if between wave dates 
	replace r`w1'date_mchg1 = h`w1'date_movein ///
		if `cond' & inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1 ///
		if `cond' & inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)

	* b) then if still missing, try date_newmarr if between wave dates 
	replace r`w1'date_mchg1 = r`w1'date_newmarr ///
		if r`w1'date_mchg1==. & `cond' & inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1 ///
		if r`w1'mchg1==0 & `cond' & inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate)

	* c) then if both out of bounds (or oob & missing), use midpoint of waves 
	replace r`w1'date_mchg1 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg1==. & `cond' & !inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg1 = 1  ///
		if r`w1'mchg1==0  & `cond' & !inrange(r`w1'date_newmarr,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_movein,t`w0'iwdate,t`w1'iwdate)
									   
	* not sweating the .f for this ... so few 
}

* RULE 3: BREAKUPS 
*		  current status sep/div AND 
* 	      prior status married/partnered AND 
*		  date_breakup<. | date_divwid<.
* -- first try date_breakup if between wave dates 
* -- then if still missing, try date_divwid if between wave dates 
* -- then if both out of bounds (or oob & missing), use midpoint of waves 
forvalues w1=14/16 {
	*local w1 14 
	local w0 = `w1'-1
	cap drop r`w1'date_mchg2
	cap drop r`w1'mchg2
	gen r`w1'date_mchg2=. 
	gen r`w1'mchg2 = 0 if t`w1'iwwave==1 
	local cond "inrange(r`w1'mstat,4,8) & inrange(r`w0'mstat,1,3) & (h`w1'date_breakup<. | r`w1'date_divwid<.) & t`w0'iwdate<. & t`w1'iwdate<."

	* a) first try date_breakup if between wave dates 
	replace r`w1'date_mchg2 = h`w1'date_breakup ///
		if `cond' & inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1 ///
		if `cond' & inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)

	* b) then if still missing, try date_divwid if between wave dates 
	replace r`w1'date_mchg2 = r`w1'date_divwid ///
		if r`w1'date_mchg2==. & `cond' & inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1 ///
		if r`w1'mchg2==0 & `cond' & inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate)

	* c) then if both out of bounds (or oob & missing), use midpoint of waves 
	replace r`w1'date_mchg2 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg2==. & `cond' & !inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1  ///
		if r`w1'mchg2==0  & `cond' & !inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate) ///
								   & !inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
								   
	* d) then if at least one is .f 
	replace r`w1'date_mchg2 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg2==. & inrange(r`w1'mstat,4,8) & inrange(r`w0'mstat,1,3) ///
		 & (h`w1'date_breakup==.f | r`w1'date_divwid==.f) & t`w0'iwdate<. & t`w1'iwdate<.
		
	replace r`w1'mchg2 = 1  ///
		if r`w1'mchg2==0 & inrange(r`w1'mstat,4,8) & inrange(r`w0'mstat,1,3) ///
		 & (h`w1'date_breakup==.f | r`w1'date_divwid==.f) & t`w0'iwdate<. & t`w1'iwdate<.
	
	format r`w1'date_mchg2 %tm
}

* redo the above for skip-wave 
forvalues w1=15/16 {
	*local w1 14 
	local w_ = `w1'-1
	local w0 = `w1'-2
	local cond "inrange(r`w1'mstat,4,8) & inrange(r`w0'mstat,1,3) & (h`w1'date_breakup<. | r`w1'date_divwid<.) & t`w0'iwdate<. & t`w1'iwdate<. & t`w_'iwdate==."

	* a) first try date_breakup if between wave dates 
	replace r`w1'date_mchg2 = h`w1'date_breakup ///
		if `cond' & inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1 ///
		if `cond' & inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)

	* b) then if still missing, try date_divwid if between wave dates 
	replace r`w1'date_mchg2 = r`w1'date_divwid ///
		if r`w1'date_mchg2==. & `cond' & inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1 ///
		if r`w1'mchg2==0 & `cond' & inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate)

	* c) then if both out of bounds (or oob & missing), use midpoint of waves 
	replace r`w1'date_mchg2 = round((t`w0'iwdate+t`w1'iwdate)/2) ///
		if r`w1'date_mchg2==. & `cond' & !inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate) ///
									   & !inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
	replace r`w1'mchg2 = 1  ///
		if r`w1'mchg2==0  & `cond' & !inrange(r`w1'date_divwid,t`w0'iwdate,t`w1'iwdate) ///
								   & !inrange(h`w1'date_breakup,t`w0'iwdate,t`w1'iwdate)
	* not sweating the .f for this ... so few 
}

* define overall marital change date 
forvalues w=14/16 {
	cap drop r`w'date_mchg
	cap drop r`w'mchg
	gen r`w'date_mchg=max(r`w'date_mchg1,r`w'date_mchg2) 
	gen r`w'mchg = max(r`w'mchg1,r`w'mchg2) if t`w'iwwave==1
	
	format r`w'date_mchg %tm
}

* ------------------------------ C_R: Health ---------------------------------* 
forvalues yr=16(2)22 { 
	*local yr 16
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'C_R, clear 
	
	/* health -- leaving these out for now bc don't care about them in 2022 
	xc005 high blood pressure 
	xc010 diabetes
	xc018 cancer of any kind excluding skin
	xc030 lung disease
	xc036 heart condition */
	* leaving these out for now 
	
	if `yr'==22 {
		gen r`rw'vax=sc327==1 if sc327<.
		gen r`rw'covid_pos = sc331==1 if sc331<.
	}
	
	local keep22 "" 
	if `yr'==22 local keep22 "r`rw'vax r`rw'covid_pos"
	keep hhid pn `keep22'
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid pn using "`temp'"
	drop if _merge ==2 
	drop _merge 
}


* ------------------------------ C_COV: COVID for 2020 ---------------------------------* 
foreach yr in 20 { 
	local yr 20
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'COV_R, clear 
	
	keep hhid pn rcovw550 rcovw551 rcovw601 rcovw602 rcovw603 rcovw605 ///
		rcovw606m1 rcovw606m2 rcovw606m3 rcovw606m4 rcovw606m5 rcovw608 rcovw609 rcovw610 rcovw611 rcovmode 

	*  covid concern
	gen r`rw'covid_concern = rcovw550 if inrange(rcovw550,1,10)
	
	*  have you had covid
	gen r`rw'covid_pos = inrange(rcovw551,1,2) if rcovw551<.
	
	*  work affected
	gen r`rw'covwk_affect = rcovw601==1 if rcovw601<6
	
	* from the above, indicate if working when covid hit 
	gen r`rw'covwk = rcovw601<6 if rcovw601<.
	
	*  stop work entirely
	gen r`rw'covwk_stop = rcovw602==1 if rcovw602<.
	
	* what happened to job (note this is only asked of those who had to stop work entirely as determined above)
	gen r`rw'covwk_event = rcovw603 if rcovw603<8
	label define covwk_event 1 "LOST JOB/LAID OFF PERMANENTLY" ///
							 2 "FURLOUGHED/LAID OFF TEMPORARILY" ///
							 3 "QUIT" ///
							 4 "OTHER" ///
							 5 "Retired" 
	label values r`rw'covwk_event covwk_event
	
	*  find new job
	gen r`rw'covwk_findnew = rcovw605==1 if rcovw605<.
	
	*  how work was affected -- primary effect 
	gen r`rw'covwk_howaffect = rcovw606m1 if rcovw606m1<8
    label define howaffect 1 "HAD TO CHANGE WORK DAYS OR HOURS" ///
						   2 "WORK BECAME MORE RISKY OR DANGEROUS" ///
						   3 "WORK BECAME HARDER" ///
						   4 "SWITCHED TO WORKING FROM HOME OR WORKING REMOTELY" ///
						   7 "OTHER" 
    label values r`rw'covwk_howaffect howaffect
	
	* work more risky/dangerous 
	gen r`rw'covwk_risk = rcovw606m1==2 | rcovw606m2==2 | rcovw606m3==2 | rcovw606m4==2 | rcovw606m5==2 if rcovw606m1<.
	
	* work got harder 
	gen r`rw'covwk_hard = rcovw606m1==3 | rcovw606m2==3 | rcovw606m3==3 | rcovw606m4==3 | rcovw606m5==3 if rcovw606m1<.
	
	* switched to wfh 
	gen r`rw'covwk_wfh = rcovw606m1==4 | rcovw606m2==4 | rcovw606m3==4 | rcovw606m4==4 | rcovw606m5==4 if rcovw606m1<.
	
	*  own/partner in business
	gen r`rw'covwk_ownbiz = rcovw608==1 if rcovw608<.
	
	*  work affected
	gen r`rw'covwk_ownaffect = rcovw609==1 if rcovw609<.
	
	*  close business
	gen r`rw'covwk_ownclose = rcovw610==1 if rcovw610<.
	
	*  permanent or temp close
	gen r`rw'covwk_ownpermclose = rcovw611==1 if rcovw611<.

	* indicate if in covid module 
	gen r`rw'covid_module = rcovmode<.
	
	keep hhid pn r15covid_concern r15covid_pos r15covwk_affect r15covwk r15covwk_stop ///
		 r15covwk_event r15covwk_findnew r15covwk_howaffect r15covwk_risk r15covwk_hard r15covwk_wfh ///
		 r15covwk_ownbiz r15covwk_ownaffect r15covwk_ownclose r15covwk_ownpermclose r15covid_module
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid pn using "`temp'"
	drop if _merge ==2 
	drop _merge 
}



* ------------------------------ H_H: Housing ---------------------------------* 
forvalues yr=16(2)22 { 
	*local yr 16
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'H_H, clear 
	keep hhid `w'subhh `w'h001 `w'h002 `w'h004 // mortgage stuff: `w'h024m1 `w'h024m2 `w'h024m3 `w'h024m4 
	
	* rename subhh 
	rename `w'subhh t`rw'subhh
	
	* farm? 
	gen h`rw'farm = `w'h001==1 if `w'h001<.
	
	* mobilehome 
	gen h`rw'mobile = `w'h002==1 if `w'h002<.
	
	* mobile or farm 
	gen h`rw'mobfarm = h`rw'farm==1 | h`rw'mobile==1
	
	* indicate if rent 
	gen h`rw'rent = `w'h004==2 if `w'h004<.
	replace h`rw'rent = 0 if h`rw'mobfarm==1
	
	tab h`rw'rent h`rw'mobfarm, missing
	
	keep hhid t`rw'subhh h`rw'rent
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid t`rw'subhh using "`temp'"
	drop if _merge ==2 
	drop _merge 
}

forvalues w=13/15 {
	replace h`w'rent=0 if inlist(t`w'nurshm,1,3)
	replace h`w'rent=0 if h`w'ahous>0 & h`w'ahous<.
}

replace h16rent=0 if h15ahous>0 & h15ahous<. & t16iwwave==1 

* still lots of missings but oh well ... 



* ------------------------------ H_J: Employment ---------------------------------* 
forvalues yr=16(2)22 { 
	*local yr 20
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)
	
	* covid questions in 22 only 
	local covvars "" 
	if `yr'==22 local covvars "sj986 sj987m1 sj987m2 sj987m3 sj987m4"

	use H`yr'J_R, clear 
	keep hhid pn `w'subhh `w'j005m1 `w'j005m2 `w'j005m3 `w'j020 /// emp questions 1
		`w'j045 `w'j021 `w'j172 `w'j517 `w'j020  /// emp questions 2
		`w'j007 `w'j008 `w'j011 `w'j012 `w'j017 `w'j018  /// events 1
		`w'j023 `w'j024 `w'j248 `w'j249 `w'j063 `w'j064 /// events 2
		`w'j595 `w'j596 `w'jw776* `w'jw777* `w'jw778* /// between-job work 
		`w'j073m1 `w'j073m2 `w'j073m3 `w'j073m4  `w'j607m1 `w'j607m2 `w'j607m3 `covvars' // reason leave questions 

	rename `w'subhh t`rw'subhh
	
	* working for pay 
	gen r`rw'workpay = `w'j020==1 if `w'j020<.
	
	* still previous employed (note this is also an HRS var)
	gen r`rw'samemp_ = inlist(`w'j045,1,3) /*& r`rw'workpay==1*/ if `w'j045<. 
	
	* self emp (rand var)
	gen r`rw'slfemp_ = `w'j021==2 if `w'j021<.
	
	* hours (rand var)
	gen r`rw'hours_ = `w'j172 if `w'j172<998 & `w'j172>-1
	
	* trying to find work 
	gen r`rw'findwork = `w'j517==1 if `w'j517<.
	
	
	* --------------- define lbrf2 ---------------* 
	gen r`rw'lbrf2 = . 
	
	* employed: any employement in j005 or workpay=1 
	replace r`rw'lbrf2 = 1 if `w'j005m1==1 | `w'j005m2==1 | `w'j005m3==1 | r`rw'workpay==1 // employed 
	
	* unemployed: if emp not already detected and evidence of unem somewhere else; searching 
	replace r`rw'lbrf2 = 2 if r`rw'lbrf2!=1 & ///
		(inrange(`w'j005m1,2,3) | inrange(`w'j005m2,2,3) | inrange(`w'j005m2,2,3)) & r`rw'findwork==1
	
	* retired: not emp/unem and retirement mentioned somewhere (ret trumps disable,homemaker,etc)
	replace r`rw'lbrf2 = 3 if !inrange(r`rw'lbrf2,1,3) & (`w'j005m1==5 | `w'j005m2==5 | `w'j005m3==5)
	
	* disabled: not any of the above and mentions disabled (trumps other and homemaker)
	replace r`rw'lbrf2 = 4 if !inrange(r`rw'lbrf2,1,5) & (`w'j005m1==4 | `w'j005m2==4 | `w'j005m3==4)
	
	* nilf: not any of the above
	replace r`rw'lbrf2 = 5 if !inrange(r`rw'lbrf2,1,6) & `w'j005m1<. 
	
	* replace as unemployed if trying to find work 
	replace r`rw'lbrf2 = 2 if inrange(r`rw'lbrf2,3,5) & r`rw'findwork==1
	
	label define lbrf2 1 "employed" 2 "unemployed" 3 "retired" 4 "disabled" 5 "nilf"
	label define j005 1 "working" 2 "unem/looking" 3 "temp layoff" 4 "disabled" 5 "retired" 6 "homemaker" 7 "other"
	label values r`rw'lbrf2 lbrf2 
	label values `w'j005m1 j005 

	
	* --------------- events -------------- *
	* date unem 
	gen r`rw'date_unem = ym(`w'j008,`w'j007) if inrange(`w'j007,1,12) & inrange(`w'j008,1900,2024)
	
	* date laid off  
	gen r`rw'date_layoff = ym(`w'j012,`w'j011) if inrange(`w'j011,1,12) & inrange(`w'j012,1900,2024)
	
	* date retired  
	gen r`rw'date_retired = ym(`w'j018,`w'j017) if inrange(`w'j017,1,12) & inrange(`w'j018,1900,2024)
	
	* date self-end   
	gen r`rw'date_selfend = ym(`w'j024,`w'j023) if inrange(`w'j023,1,12) & inrange(`w'j024,1900,2024)
	
	* date stopped working prev 
	gen r`rw'date_leaveemp = ym(`w'j064,`w'j063) if inrange(`w'j063,1,12) & inrange(`w'j064,1900,2024)
	
	* date start job    
	gen r`rw'date_startjob = ym(`w'j249,`w'j248) if inrange(`w'j248,1,12) & inrange(`w'j249,1900,2024)
	
	foreach var in r`rw'date_unem r`rw'date_layoff  r`rw'date_retired r`rw'date_selfend ///
				   r`rw'date_leaveemp r`rw'date_startjob { 
		format `var' %tm 
	}
	
	* --------------- reasons 1 -------------- *
	gen r`rw'whyleave = . 
	replace r`rw'whyleave = `w'j073m1 if inrange(`w'j073m1,1,26)
	replace r`rw'whyleave = `w'j073m2 if r`rw'whyleave==. & inrange(`w'j073m2,1,26)
	replace r`rw'whyleave = `w'j073m3 if r`rw'whyleave==. & inrange(`w'j073m2,1,26)
	
	label define whyleave ///
		    1 "BIZ CLOSED" ///
            2 "LAID OFF" ///
            3 "HEALTH/DISABLED" ///
            4 "FAMILY CARE" ///
            5 "BETTER JOB" ///
            6 "QUIT" ///
            7 "RETIRED" ///
            8 "Fam moved" ///
            9 "OWNERSHIP CHANGED" ///
           10 "PENSION RULES CHANGED" ///
		   11 "COVID concern" ///
           14 "Divorce/Separation" ///
           15 "Passed to family members" ///
           16 "Transportation" ///
           23 "Travel" ///
           24 "Early ret offer" ///
           25 "Tax/SSA" ///
		   26 "COVID"
	label values r`rw'whyleave whyleave 
	
	* for 2022, replace reason why left as covid if indicated in covid q 
	if `yr'==22 {  
		replace r`rw'whyleave = 11 if `w'j986==1
	}
	
	* alternate covid why leave -- if any covid reason given 
	gen r`rw'whyleave_covid = 0 if `w'j073m1<. | `w'j073m2<. | `w'j073m3<.
	replace r`rw'whyleave_covid = 1 if inlist(`w'j073m1,11,26) | inlist(`w'j073m2,11,26) | inlist(`w'j073m3,11,26) 
	if `yr'==22 {  
		replace r`rw'whyleave_covid = 1 if `w'j986==1
	}
	
	* --------------- reasons 2 (for iw work desc'd below) -------------- *
	gen r`rw'whyleave_btw = . 
	replace r`rw'whyleave_btw = `w'j607m1 if inrange(`w'j607m1,1,26)
	replace r`rw'whyleave_btw = `w'j607m2 if r`rw'whyleave_btw==. & inrange(`w'j607m2,1,26)
	replace r`rw'whyleave_btw = `w'j607m3 if r`rw'whyleave_btw==. & inrange(`w'j607m2,1,26)
	label values r`rw'whyleave_btw whyleave 
	
	* alternate covid why leave -- if any covid reason given 
	gen r`rw'whyleave_btw_covid = 0 if `w'j607m1<. | `w'j607m2<. | `w'j607m3<.
	replace r`rw'whyleave_btw_covid = 1 if inlist(`w'j607m1,11,26) | inlist(`w'j607m2,11,26) ///
		  | inlist(`w'j607m3,11,26) 
		  
	* --------------- whyleave combined (both kinds of job) ---------------- * 
	gen r`rw'whyleave_any = . 
	replace r`rw'whyleave_any = r`rw'whyleave if r`rw'whyleave<.
	replace r`rw'whyleave_any = r`rw'whyleave_btw if r`rw'whyleave_btw<. & r`rw'whyleave_any==. 
	label values r`rw'whyleave_any whyleave 
	
	gen r`rw'whyleave_any_covid = . 
	replace r`rw'whyleave_any_covid = r`rw'whyleave_covid if r`rw'whyleave_covid<.
	replace r`rw'whyleave_any_covid = r`rw'whyleave_btw_covid if r`rw'whyleave_btw_covid<. ///
			& r`rw'whyleave_any_covid==. 
		  
	* ----------------------- between-iw work ------------------ *
	* this is a bit complicated. respondents might have work/jobs between waves that 
	* are the jobs they're doing during the waves. the presence of these inter-wave 
	* jobs are recorded in the questions referenced below. HRS tries to identify 
	* exactly which months are worked in each possible inter-wave year. I capture them 
	* as monthly dates in the r`rw'btwwork_yY_mM vars below, where Y and M are just 
	* place holder Y(1,4) and M(1,12); the M here is NOT a month, it's just the Mth
	* possible month the person listed. 
	
	* work between iw jobs 
	gen r`rw'btwjob = `w'j596==1 if `w'j596<.
	
	* loop through question groups 17-20 
	local ynum = 4
	forvalues q=17/20 {
		
		* whether any work in year ___ 
		gen r`rw'anybtwwrk_y`ynum' = `w'jw777_`q'==1 if `w'jw777_`q'<.
		
		* loop through possible months 
		forvalues m=1/12 {  
			cap confirm variable `w'jw778_`q'm`m'
			if _rc == 0 {
				gen r`rw'btwwork_y`ynum'_m`m' = ym(`w'jw776_`q',`w'jw778_`q'm`m') if inrange(`w'jw778_`q'm`m',1,12)
				format r`rw'btwwork_y`ynum'_m`m' %tm
			} 
			else {
				display "Variable `w'jw778_`q'm`m' does not exist"
			}
		}
		local ynum = `ynum'-1
	}
	
	keep hhid pn r`rw'workpay r`rw'samemp_ r`rw'slfemp_ r`rw'hours_ r`rw'findwork r`rw'lbrf2 ///
		 r`rw'date_unem r`rw'date_layoff  r`rw'date_retired r`rw'date_selfend r`rw'date_leaveemp r`rw'date_startjob ///
		 t`rw'subhh r`rw'whyleave* r`rw'btwjob r`rw'anybtwwrk* r`rw'btwwork_*
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid pn t`rw'subhh using "`temp'"
	drop if _merge ==2 
	drop _merge 
}

* replace 16s 
gen r16jhours = r16hours_ 
gen r16slfemp = r16slfemp_ 
gen r16samemp = r16samemp_

drop r*hours_ r*slfemp_ r*samemp_


* ------------------------------ H_J3: Retirement ---------------------------------* 
forvalues yr=16(2)22 { 
	*local yr 16
	frame2 core, replace 
	waveyr `yr'
	local year = r(year)
	local w = r(w)
	local rw = r(rw)

	use H`yr'J3_R, clear 
	
	
	keep hhid pn `w'subhh `w'j3578 `w'j3580 `w'j3581 `w'j3583 `w'j3584 ///	
		 `w'j3588_1  `w'j3588_2 `w'j3588_3 `w'j3588_4 

	* say emp 
	gen r`rw'sayret_ = . 
	replace r`rw'sayret_ = 0 if `w'j3578==5
	replace r`rw'sayret_ = 1 if `w'j3578==1
	replace r`rw'sayret_ = 2 if `w'j3578==3
	replace r`rw'sayret_ = 3 if `w'j3578==7
	
	* when retired -- note this includes partly retired 
	gen r`rw'date_retired2 = ym(`w'j3581,`w'j3580) if inrange(`w'j3581,1900,2024) & inrange(`w'j3580,1,12)
	format r`rw'date_retired2 %tm 
	label variable r`rw'date_retired2 "date retired (incl partial ret)"
	
	* retirement forced or voluntary? 
	gen r`rw'retforce = `w'j3583  if inrange(`w'j3583,1,3)
	label define retforce 1 "wanted" 2 "forced" 3 "partly forced"
	label values r`rw'retforce retforce
	
	* satisfied with retirement 
	gen r`rw'retsat = `w'j3584 if inrange(`w'j3584,1,3) 
	label define retsat 1 "very" 2 "moderately" 3 "not at all"
	label values r`rw'retsat retsat	
	
	* why retired 
	gen r`rw'ret_health = `w'j3588_1 if inrange(`w'j3588_1,1,4) 
	gen r`rw'ret_dothings = `w'j3588_2 if inrange(`w'j3588_2,1,4) 
	gen r`rw'ret_nowork = `w'j3588_3 if inrange(`w'j3588_3,1,4) 
	gen r`rw'ret_family = `w'j3588_4 if inrange(`w'j3588_4,1,4) 
	
	rename `w'subhh t`rw'subhh
	
	keep hhid pn t`rw'subhh r`rw'sayret_ r`rw'date_retired2 r`rw'ret_* r`rw'retforce r`rw'retsat 
	
	tempfile temp 
	save "`temp'"
	frame change default 
	merge m:1 hhid pn t`rw'subhh using "`temp'"
	drop if _merge ==2 
	drop _merge 
}

* sayrets are exactly the same --replace 16 
rename r16sayret_ r16sayret
drop *sayret_

* use sayrets to reclassify some lbrf2 (a la rand does for lbrf)
forvalues w=13/16{ 
	replace r`w'lbrf2 = 3 if inrange(r`w'lbrf2,4,5) & inrange(r`w'sayret,1,2)
}

* create retired var 
forvalues w=13/16{ 
	gen r`w'retired = r`w'lbrf2==3 if r`w'lbrf2<.
}


* ----------------------------- reshape long -----------------------------
*keep hhid pn t_birthmo t_birthyr t_degree t_gender t_hispanic t_knowndeceasedmo t_knowndeceasedyr t_race t_usborn t1*
* rename t* vars 
forvalues w=13/16 {
	rename t`w'* *_`w'
}

* rename r* vars 
forvalues w=13/16 {
	rename r`w'* *_`w'
}

* rename h* vars 
forvalues w=13/16 {
	rename h`w'* hh_*_`w'
}

* rename s* vars 
forvalues w=13/15 {
	rename s`w'* sp_*_`w'
}

* create var list 
local vlist ""
foreach var of varlist _all {
	local suff = substr("`var'", -2, 2)
	local vname = substr("`var'", 1, strlen("`var'")-3 ) 
    if inlist("`suff'","13","14","15","16") {
		if strpos("`vlist'"," `vname'_ ")==0 { 
			local vlist = " `vlist' " + " `vname'_ "
		}
    }
}
di "`vlist'"

reshape long `vlist',  i(hhid pn) j(wave)

* wave as last 2 dig of year 
*replace wave = wave*2-10

* get rid of underscore added for reshape 
rename *_ *

* don't know where these came from, mostly blank 
drop if pn==""

* xtset 
xtset hhidpn wave

* little wave indicator to help with viewing 
gen w_ = ""
replace w_="===" if wave==13
order w_ *
format w_ %3s


* ------------------------ cleaning employment transitions ---------------------
frame change default 

* employment change 
cap program drop empchg_define
program define empchg_define
	cap drop empchg 
	cap drop empchg_any
	gen  empchg = (l.lbrf2!=lbrf2 & l.lbrf2<. & lbrf2<.) /// 
			   | (l2.lbrf2!=lbrf2 & l2.lbrf2<. & l.lbrf2==. & lbrf2<.) ///
			   | (l3.lbrf2!=lbrf2 & l3.lbrf2<. & l2.lbrf2==. & l.lbrf2==. & lbrf2<.) // 
			   
	* indicate if any empchg 
	bys hhidpn (wave): egen empchg_any = max(empchg)
end
empchg_define
			   

* ----------- try to capture spurious emp changes 
* identify whether always ret/nilf/dis 
gen lbrf_rnd = inrange(lbrf2,3,5)
bys hhidpn (wave): egen lbrf_rnd_min = min(lbrf_rnd)

* get max date_retired 
bys hhidpn (wave): egen date_retired_max = max(date_retired)
format date_retired_max %tm

bys hhidpn (wave): egen date_retired2_max = max(date_retired2) // recall this one includes partial 
format date_retired2_max %tm

* nilf/dis in some wave but evidence elsewhere of an earlier ret (v1)
gen nilfret_flag = inrange(lbrf2,4,5) & iwdate>date_retired_max & date_retired_max<. & ///
	(f.lbrf2==3 | f2.lbrf2==3 | f3.lbrf2==3) 
	
* nilf/dis in some wave but evidence elsewhere of an earlier ret (v2 -- requires always rnd to avoid partial) 
gen nilfret2_flag = inrange(lbrf2,4,5) & iwdate>date_retired2_max & date_retired2_max<. & ///
	(f.lbrf2==3 | f2.lbrf2==3 | f3.lbrf2==3) & lbrf_rnd_min==1

* whether goes from ret to nilf/dis and back to ret 
gen rnr_flag = (l2.lbrf2==3 | l.lbrf2==3) ///	
			 & inrange(lbrf2,4,5) ///
			 & (f.lbrf2==3 | f2.lbrf2==3)

* id whether R->N/D and always lbrf_rnd 
gen ret_nd_flag = lbrf_rnd_min==1 & inrange(lbrf2,4,5) & (l.lbrf2==3 | l2.lbrf2==3 | l3.lbrf2==3)

* flag any of these 
gen anyretflag = max(nilfret_flag, nilfret2_flag,rnr_flag,ret_nd_flag)

* lbrf2_orig and new lbrf2 
rename lbrf2 lbrf2_orig
gen lbrf2 = lbrf2_orig 
label values lbrf2 lbrf2
replace lbrf2=3 if anyretflag==1

* redefine empchg 
empchg_define

*---------------------------- transitions prep --------------------------------- * 
* emp->nonemp: indicate if valid date given and what it is 
cap drop date_empchg
gen date_empchg = . 
foreach date in date_unem date_layoff date_retired date_selfend /// from employment 
			    date_leaveemp date_retired2 { 
	replace date_empchg = `date' if empchg==1 & ///
		((inrange(`date',l.iwdate,iwdate) &    lbrf2!=l.lbrf2  & l.lbrf2 !=. & lbrf2!=1) 	/// no skip 
		 | (inrange(`date',l2.iwdate,iwdate) & lbrf2!=l2.lbrf2 & l2.lbrf2!=. & l.lbrf2==. & lbrf2!=1)  /// one skip  
		 | (inrange(`date',l3.iwdate,iwdate) & lbrf2!=l3.lbrf2 & l3.lbrf2!=. & l.lbrf2==. & l2.lbrf2==. & lbrf2!=1)) // two skip 
}

foreach date in date_startjob { 
	replace date_empchg = `date' if empchg==1 & ///
		((inrange(`date',l.iwdate,iwdate) &    lbrf2!=l.lbrf2  & l.lbrf2 !=. & lbrf2==1) 	/// no skip 
		 | (inrange(`date',l2.iwdate,iwdate) & lbrf2!=l2.lbrf2 & l2.lbrf2!=. & lbrf2==1) /// one skip  
		 | (inrange(`date',l3.iwdate,iwdate) & lbrf2!=l3.lbrf2 & l3.lbrf2!=. & lbrf2==1)) // two skip 
}

format date_empchg %tm
gen any_date_empchg  = date_empchg<.

gen flag_date = "" 
replace flag_date = "==" if empchg==1 & any_date_empchg==0

* create intermediate dates if none valid provided 
gen date_empchg_i = . 
replace date_empchg_i = round((iwdate+l.iwdate)/2) if  empchg==1 & date_empchg==.
replace date_empchg_i = round((iwdate+l2.iwdate)/2) if empchg==1 & date_empchg==. & l.iwdate==.
replace date_empchg_i = round((iwdate+l3.iwdate)/2) if empchg==1 & date_empchg==. & l.iwdate==. & l2.iwdate==. 

* make sure every empchg has date associated with it 
assert (date_empchg<. | date_empchg_i<.) if empchg==1

format date_empchg_i %tm

* create flag for date_empchg_i then make all date_empchg
gen date_empchg_flag = date_empchg_i<. 
replace date_empchg = date_empchg_i if date_empchg==. & date_empchg_i<.

* move date_empchg back if covid reason given but date_empchg<mar2020 (could be bad impute or confused recall)
gen date_empchg_covid_flag = date_empchg<`=tm(2020m3)' & inlist(whyleave_any,11,26)
replace date_empchg = round((iwdate+`=tm(2020m3)')/2) if date_empchg_covid_flag==1
count if date_empchg<`=tm(2020m3)' & inlist(whyleave_any,11,26)
drop date_empchg_covid_flag

* move btw wave emp indicators 
foreach var of varlist _all { 
	if substr("`var'",1,3)=="btw" {
		replace `var' = `var'[_n+1] if hhidpn[_n+1]==hhidpn
	}
}

order w_ wave iwdate lbrf2 *

save hrs_temp, replace

* -------------------------------------------------------------------------- * 
* ------------------------ create monthly record --------------------------- * 
* -------------------------------------------------------------------------- * 
frame change default
use hrs_temp, clear 

frame copy default monthly, replace 
frame change monthly
drop if iwdate==.

keep hhidpn wave iwdate lbrf2 empchg_any /*empchg date_empchg*/ alive btw* /// 
	mstat /*mchg date_mchg*/ t_knowndeceasedmo t_knowndeceasedyr
*order hhidpn wave iwdate lbrf2 empchg empchg_any date_empchg alive mstat mchg date_mchg btw* *

* death date 
gen date_death = ym(t_knowndeceasedyr,t_knowndeceasedmo)
format date_death %tm
drop t_knowndeceasedyr t_knowndeceasedmo

* get lbrf, mstat next wave 
gen lbrf2_nw = lbrf2[_n+1] if hhidpn[_n+1]==hhidpn 
label values lbrf2_nw lbrf2

gen mstat_nw = mstat[_n+1] if hhidpn[_n+1]==hhidpn 
label values mstat_nw mstat


*--------------- expand and make monthly var ---------------
* get number of months between waves 
gen num_mo = iwdate[_n+1]-iwdate if hhidpn[_n+1]==hhidpn
replace num_mo = 1 if num_mo==.

expand num_mo, gen(w_)
sort hhidpn wave

gen mo = iwdate 
bys hhidpn wave (w_): replace mo = mo + _n - 1
replace w_ = 1-w_ 
label define w_ 0 "" 1 "===="
label values w_ w_
format mo %tm

*order hhidpn wave iwdate mo w_ lbrf2 lbrf2_nw empchg empchg_any date_empchg  alive 
*br    hhidpn wave iwdate mo w_ lbrf2 lbrf2_nw empchg empchg_any date_empchg  alive 

*--------------- between work  ---------------
* note: not doing anything with this as yet, need to investigate more (adds complication)
* create indicator if working that month from btw work indicators 
gen work_btw = 0 
forvalues y=1/4 {
	forvalues m=1/12 {
		cap replace work_btw = 1 if mo==btwwork_y`y'_m`m' 
	}
}

* indicate if any between work for hhidpn 
bys hhidpn (mo): egen any_work_btw = max(work_btw)

*--------------- finalize work and mstat indicators ---------------
* -------create date_empchg collapsed dataset and merge on hhidpn mo ---------
frame2 empchg, replace 
use hrs_temp, clear 
keep hhidpn date_empchg date_empchg_flag empchg whyleave whyleave_covid whyleave_btw whyleave_btw_covid whyleave_any whyleave_any_covid lbrf2 wave iwdate
tab  date_empchg empchg, mi // to ensure that all empchanges have dates and vice versa 
keep if empchg==1

* deal with a few problem obs where same date has two transitions (reported in neighboring waves)
* solution: move later-reported date back one month
duplicates tag hhidpn date_empchg, gen(dup) 
bys hhidpn: gen n=_n 
replace date_empchg = date_empchg+1 if dup==1 & n>1 

gen mo = date_empchg 
drop lbrf2 wave iwdate n dup 
tempfile empchg 
save "`empchg'"
frame change monthly 
merge 1:1 hhidpn mo using "`empchg'"
assert _merge!=2 // all emp changes rematched 
drop _merge 
frame drop empchg

* create "final" lbrf incorporating between-wave employment changes (recorded or imputed)
bys hhidpn wave: egen sumchg = sum(empchg) // bc some wvs have mult changes (reported in neighboring waves)
gen lbrff = . 
bys hhidpn wave: replace lbrff=lbrf2 if _n==1
replace lbrff=lbrf2_nw if mo==date_empchg & lbrf2_nw<. & !(date_empchg==iwdate & sumchg==2)
bys hhidpn: replace lbrff=lbrff[_n-1] if lbrff==.

label values lbrff lbrf2

xtset hhidpn mo

* -------create date_mchg collapsed dataset and merge on hhidpn mo ---------
* create "final" mstat incorporating between-wave employment changes (recorded or imputed)
frame2 mchg, replace 
use hrs_temp, clear 
keep hhidpn date_mchg mchg mstat wave iwdate
tab  date_mchg mchg, mi // to ensure that all empchanges have dates and vice versa 
keep if mchg==1

* unlike empchg, no duplicates here 
duplicates report hhidpn date_mchg, gen(dup)    
assert r(unique_value) == r(N)

gen mo = date_mchg 
drop mstat wave iwdate 
tempfile mchg 
save "`mchg'"
frame change monthly 
merge 1:1 hhidpn mo using "`mchg'"
assert _merge!=2 // all emp changes rematched 
drop _merge 
frame drop mchg

* create "final" mstat incorporating between-wave marital changes (recorded or imputed)
bys hhidpn wave: egen summchg = sum(mchg) // to mirror what was done for empchg ... 
tab summchg 							  // ...tho not necessary 
gen mstatf = . 
bys hhidpn wave: replace mstatf=mstat if _n==1
replace mstatf=mstat_nw if mo==date_mchg & mstat_nw<. & !(date_mchg==iwdate & summchg==2)
bys hhidpn: replace mstatf=mstatf[_n-1] if mstatf==.
label values mstatf MSTATUS


order hhidpn wave iwdate mo w_ lbrf2 lbrf2_nw lbrff empchg empchg_any date_empchg* mstat mstat_nw mstatf mchg date_mchg
* br    hhidpn wave iwdate mo w_ lbrf2 lbrf2_nw lbrff empchg empchg_any date_empchg*  // look at emp changes 
* br    hhidpn wave iwdate mo w_ mstat mstat_nw mstatf mchg date_mchg // look at mchanges 
* br    hhidpn wave iwdate mo w_ lbrff mstatf date_death 

* bring out the dead 
gen dead = 0 
replace dead = 1 if inrange(alive,5,6)
replace dead = 1 if mo>=date_death

* keep only essential vars 
keep hhidpn wave iwdate mo w_ lbrff mstatf dead empchg mchg date_empchg* date_mchg whyleave* date_death


* ----------------------------------------------------------------------------- * 
* -------------------------- Bring covars in  -------------------------- * 
* ----------------------------------------------------------------------------- * 
frame copy default covars, replace 
frame change covars

keep hhid pn wave t_birthmo t_birthyr t_degree t_gender t_hispanic t_race t_usborn ///
	 nurshm ppn subhh wgtr hhidpn ragender raracem raeduc rabplace cendiv urbrur ///
	 hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap ///
	 /// date_empchg_flag whyleave whyleave_covid whyleave_btw whyleave_btw_covid whyleave_any whyleave_any_covid ///
	 btwjob sayret rplnya ret_health ret_dothings ret_nowork ret_family retforce retsat ///
	 shlt hltc jhours slfemp samemp jcten jcoccc jcindc jloccc jlindc wgihr wgiwk fsize union jcpen /// 
	 covid_pos covid_concern covwk_affect covwk covwk_stop covwk_event covwk_findnew covwk_howaffect ///
	 covwk_risk covwk_hard covwk_wfh covwk_ownbiz covwk_ownaffect covwk_ownclose covwk_ownpermclose covid_module
	 
	// sp_shlt sp_hltc3 sp_sayret sp_slfemp sp_retemp sp_lbrf sp_jhours sp_jcten sp_jcocc sp_jcind ///
	// sp_jlasty sp_jlastm sp_samemp sp_wgihr sp_wgiwk sp_inlbrf sp_fsize sp_union 

* gender var 
tab t_gender ragender, m // identical 
gen gender = ragender-1
label define gender 0 "male" 1 "female"
label values gender gender
drop t_gender ragender

* race+ var 
tab t_race raracem, m
gen race = raracem 
replace race=4 if race==3
replace race=3 if inrange(t_hispanic,1,3)
label define race 1 "white NH" 2 "black NH" 3 "hisp" 4 "other"
label values race race 
drop t_race t_hispanic raracem

* educ 
tab t_degree raeduc, m // use rand 
rename raeduc educ 
drop t_degree 

* birth date 
gen bdate = ym(t_birthyr, t_birthmo)
format bdate %tm 
drop t_birthmo t_birthyr

* nativity 
gen foreign = t_usborn==5 
drop rabplace t_usborn 

* nursing home -- indicate only the certainly-nursing-home responses
replace nurshm = inlist(nurshm,1,3)

order hhidpn hhid pn ppn subhh wgtr wave gender race bdate foreign educ nurshm cendiv urbrur ///
	hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap ///
	 btwjob  sayret rplnya ret_health ret_dothings ret_nowork ret_family retforce retsat ///
	 shlt hltc jhours slfemp samemp jcten jcocc jcind jloccc jlindc wgihr wgiwk fsize union jcpen *

* merge to monthly  
tempfile covars 
save "`covars'"
frame change monthly 
merge m:1 hhidpn wave using "`covars'"
assert _merge>1
drop if _merge==2
drop _merge

xtset hhidpn mo

frame drop covars


* -----------------------------------------------------------------------------* 
* ---------------------------- fixing vars ------------------------------------* 
* -----------------------------------------------------------------------------* 
*indicate number of missing 
* misstable sum hhidpn wave iwdate mo w_ lbrff mstatf dead hhid pn ppn subhh wgtr gender race bdate foreign educ nurshm cendiv urbrur hh_numch hh_numad hh_rent hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap whyleave whyleave_btw btwjob sayret ret_health ret_dothings ret_nowork ret_family shlt hltc3 jhours slfemp samemp jcten jcocc jcind

* define age 
gen age = floor((mo-bdate)/12) 
drop bdate

* Rand vars in wave 16 to fill with those from prev wave 
foreach var in cendiv urbrur hh_hhres hh_child hh_achck hh_ahous hh_amort hh_atoth hh_atotf hh_icap shlt { 
	replace `var' = l.`var' if wave==16
}

* impute urban-rural (TBD)
replace cendiv=.e if cendiv==11
tab cendiv urbrur if dead==0 & iwdate==mo, m


* employment vars 
* NOTE: issue is about changes in employment status -- need to import values from prior/subseq waves 
* misstable sum jhours slfemp samemp jcten jcocc jcind wgihr wgiwk fsize union jcpen if lbrf==1


* -------------------------------- fix employment covars  -------------------------------
* copy to other frame 
frame copy monthly fix, replace 
frame change fix 

* collapse (max) employed by hhidpn, wave, employment status 
gen employed = lbrff==1
collapse (first) mo jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen empchg,  ///
	by(hhidpn wave employed)
	
* indicate whether to/from emp 
bys hhidpn wave: gen num=_N

gen mo_e0_ = (1-employed)*mo 
gen mo_e1_ = (employed)*mo 
bys hhidpn wave: egen mo_e0 = max(mo_e0_)
bys hhidpn wave: egen mo_e1 = max(mo_e1_)
drop mo_e0_ mo_e1_
gen diff = mo_e1-mo_e0

gen emptrans = 0 
replace emptrans = 1 if num==2 & diff<0 // E0 
replace emptrans = 2 if num==2 & diff>0 // 0E 
label define emptrans 0 "" 1 "from" 2 "to", replace
label values emptrans emptrans

tab emptrans empchg if employed==1 // the ones in "employed==1 & emptrans==0 & empchg==1" are likely those where date of transition is first or last date of a wave 

* keep only employed obs where both exist in wave 
drop if num==2 & employed==0
drop num diff mo* 

xtset hhidpn wave
order hhidpn wave employed empchg emptrans jhours slfemp samemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen 

* flag missing hours etc where should be something 
foreach var in jhours slfemp jcten jcoccc jcindc wgihr wgiwk fsize union jcpen  { 
	gen `var'_flag = `var'>=. & employed==1 
}

* borrowing algorithm:
foreach var in jhours slfemp jcoccc jcindc wgihr wgiwk fsize union jcpen {
	di "`var'"
	* if empchg FROM employment or other missing: borrow from earlier wave 
	gen `var'_i = . 
	replace `var'_i=l.`var' if `var'_flag==1 & l.`var'<. & emptrans==1

	* if empchg TO employment: borrow from later wave 
	replace `var'_i=f.`var' if `var'_flag==1 & f.`var'<. & emptrans==2

	* if no empchg and same emp: try to borrow from prev wave 
	replace `var'_i=l.`var' if `var'_flag==1 & l.`var'<. & emptrans==0 & samemp==1

	* if no empchg and same emp: try to borrow from subsequent wave 
	replace `var'_i=f.`var' if `var'_flag==1 & f.`var'<. & emptrans==0 & samemp==0

	* for all other, try both 
	replace `var'_i=l.`var' if `var'_flag==1 & l.`var'<. & emptrans==0 & `var'_i==.
	replace `var'_i=f.`var' if `var'_flag==1 & f.`var'<. & emptrans==0 & `var'_i==.

	
	* for remaining missings ... 
	* averages
	if inlist("`var'","jhours","wgihr","wgiwk","fsize") {
		bys hhidpn (wave): egen `var'_avg = mean(`var')
		replace `var'_i=`var'_avg if `var'_flag==1 & `var'_avg<. & `var'_i==.
		drop `var'_avg
	} 
	if inlist("`var'","slfemp","jcoccc","jcindc","union","jfpen") {
		bys hhidpn (wave): egen `var'_mode = mode(`var'), minmode
		replace `var'_i=`var'_mode if `var'_flag==1 & `var'_mode<. & `var'_i==.
		drop `var'_mode
	}
}

* ------ do it special for jcten ------
* basic changes 
gen jcten_i = . 
replace jcten_i=0 if jcten_flag==1 & jcten_i==. & samemp==0 
replace jcten_i=l.jcten+2 if jcten_flag==1 & jcten_i==. & samemp==1
replace jcten_i=l2.jcten+4 if jcten_flag==1 & jcten_i==. & samemp==1 & l.hhidpn==.

* infer from empchg trans  
replace jcten_i=l.jcten+1 if jcten_flag==1 & l.jcten<. & emptrans==1 & jcten_i==. 
replace jcten_i=l.jcten_i+1 if jcten_flag==1 & l.jcten_i<. & emptrans==1 & jcten_i==. 
replace jcten_i=max(f.jcten-1,0) if jcten_flag==1 & f.jcten<. & emptrans==2 & jcten_i==. 
replace jcten_i=max(f.jcten_i-1,0) if jcten_flag==1 & f.jcten_i<. & emptrans==2 & jcten_i==. 

* if going to employment and prior wave not employed, tenure 0 
replace jcten_i=0 if jcten_flag==1 & jcten_i==. & emptrans==2 & (l.employed==0 | (l.hhidpn==. & l2.employed==0))

* if prior is from and current is to, make tenure 0 
replace jcten_i=0 if jcten_flag==1 & jcten_i==. & emptrans==2 & (l.emptrans==1 | (l.hhidpn==. & l2.emptrans==1))

* use sameemp information again
replace jcten_i=l.jcten+2 if jcten_flag==1 & l.jcten<. & samemp==1 & jcten_i==. 
replace jcten_i=l.jcten_i+2 if jcten_flag==1 & l.jcten_i<. & samemp==1 & jcten_i==. 

* self emp -- if still, add 2 
replace jcten_i=l.jcten+2 if jcten_flag==1 & l.jcten<. & slfemp==1 & l.slfemp==1 & jcten_i==. 
replace jcten_i=l2.jcten+4 if jcten_flag==1 & l2.jcten<. & slfemp==1 & l2.slfemp==1 & jcten_i==. & l.hhidpn==.

* if going from not-self to self, tenure 0 (and vice versa)
replace jcten_i=0 if jcten_flag==1 & jcten_i==. & l.slfemp==0 & slfemp==1
replace jcten_i=0 if jcten_flag==1 & jcten_i==. & l.slfemp==1 & slfemp==0

/* testing 
local var jcten
cap drop `var'_flag_ever
ever, name(`var'_flag_ever) condition(`var'_flag==1 & `var'_i==.) maxmin(max) byvar(hhidpn) svar(wave)
count if `var'_flag==1 
count if `var'_flag==1 & `var'_i>=.
tab wave if `var'_flag==1 * `var'_i>=.
br hhidpn wave employed empchg emptrans slfemp samemp `var'* if `var'_flag_ever==1
drop `var'_flag_ever
*/

* move inferred over to real 
foreach var in slfemp jhours jcoccc jcindc wgihr wgiwk fsize union jcpen jcten {
	replace `var'_i=`var' if `var'<. 
}

* keep 
keep hhidpn wave *_flag *_i

* merge back on original and redefine original vars 
tempfile fix 
save "`fix'"
frame change monthly 
merge m:1 hhidpn wave using "`fix'"
assert _merge==3
drop _merge 

* replace missing vars 
foreach var in slfemp jhours jcoccc jcindc wgihr wgiwk fsize union jcpen jcten {
	replace `var'=`var'_i if `var'>=. & `var'_i<.
}

* -------------------------some final adjustments ------------------------------
replace union=0 if slfemp==1

* full time 
gen ft = inrange(jhours,35,200)

* financial vars 
foreach var in achck ahous amort atoth atotf icap { 
	gen l`var' = log(hh_`var'+1)
}

* indicate if any adult or minor children 
gen any_adult = hh_numad>=1 & hh_numad<. // nb shouldn't be any missings 
gen any_child = hh_numch>=1 & hh_numch<. // nb shouldn't be any missings 


drop *_i


frame drop fix 


* -------------------------------------------------------------------------- * 
* ------------------------ create spouse data  ----------------------------- * 
* -------------------------------------------------------------------------- * 
frame change monthly
cap drop sp_* _merge*
destring ppn pn, replace 

* variable list 
local vlist lbrff dead nurshm shlt ft age covid_pos 

* first attempt: on month 
frame copy monthly spouse, replace 
frame change spouse 
drop if ppn==0 
keep hhid subhh pn mo `vlist' 
rename pn ppn 
foreach var in `vlist' { 
	rename `var' sp_`var'
}
tempfile spouse 
save "`spouse'"
frame change monthly 
merge m:1 hhid subhh ppn mo using "`spouse'", gen(_merge_sp)
drop if _merge_sp==2

* second attempt: on wave
frame copy monthly spouse, replace 
frame change spouse 
drop if ppn==0 
keep hhid subhh pn wave `vlist' 
bys hhid pn wave: keep if _n==1
rename pn ppn 
foreach var in `vlist' { 
	rename `var' sp_`var'_w
}
tempfile spouse 
save "`spouse'"
frame change monthly 
merge m:1 hhid subhh ppn wave using "`spouse'", gen(_merge_spw)
drop if _merge_spw==2

sort hhid pn subhh mo

* br hhid pn ppn mo mstat lbrf sp_lbrff sp_lbrff_w

* use initial observations of spouse vars when monthly obs don't line up
foreach var in `vlist' {
	replace sp_`var' = sp_`var'_w if sp_`var'==. & sp_`var'_w<. & mstatf<=3
}
frame drop spouse
drop *_w

* ---------- drop some obs now that spouses done ------------------* 
* drop those dead for more than one month 
bys hhidpn dead (mo): gen deadmo = _n
replace deadmo = deadmo*dead
drop if deadmo>1

/* drop if always retired and/or always missing lbrff 
ever, name(retired_min) condition((lbrff==3 | lbrff==.)) maxmin(min) byvar(hhidpn) svar(mo)
drop if retired_min==1 
drop retired_min */

* drop if always dead 
ever, name(dead_min) condition(dead==1) maxmin(min) byvar(hhidpn) svar(mo)
drop if dead_min==1 
drop dead_min 


* -------------------------------------------------------------------------- * 
* --------------------------- finalize  -------------------------------------* 
* -------------------------------------------------------------------------- * 
gen covid = mo>=`=tm(2020m3)'

* sample indicators -- in wave, working in wave 
forvalues w=13/16 {
	ever, name(inw`w') condition(wave==`w') byvar(hhidpn) svar(mo) maxmin(max)
	ever, name(emp`w') condition(wave==`w' & lbrff==1) byvar(hhidpn) svar(mo) maxmin(max)
	ever, name(ret`w') condition(wave==`w' & lbrff==3) byvar(hhidpn) svar(mo) maxmin(max)
	ever, name(noret`w') condition(wave==`w' & lbrff!=3 & lbrff<.) byvar(hhidpn) svar(mo) maxmin(max)
	ever, name(died`w') condition(wave==`w' & dead==1) byvar(hhidpn) svar(mo) maxmin(max)
}

gen allw = inw13==1 & inw14==1 & inw15==1 & inw16==1 

* tag waves
egen tag_hw = tag(hhidpn wave)

* get lbrff at beginning and end of each wave 
frame copy monthly temp, replace 
frame change temp 
collapse (first) lbrff_begw=lbrf (last) lbrff_endw=lbrf , by(hhidpn wave)
tempfile temp 
save "`temp'"
frame change monthly 
merge m:1 hhidpn wave using "`temp'"
assert _merge==3 
drop _merge


* whether ever had covid, whether in covid group 
ever, name(covid_any) condition(covid_pos==1) byvar(hhidpn) svar(mo) maxmin(max)
ever, name(incovid) condition(covid_module==1) byvar(hhidpn) svar(mo) maxmin(max)


* -------------------------------------------------------------------------- * 
* ------------- create sample of covid-survey respondents ----------------------
* -------------------------------------------------------------------------- * 
frame copy monthly covidwt, replace 
frame change covidwt 
keep if mo==iwdate 

* create age groups
xtile ageg = age if wgtr>0, n(15) 
tab age ageg

* create groups based on race-age-gender and cendiv-age-gender and wave-age-gender
egen rag = group(race ageg gender)
egen dag = group(cendiv ageg gender)
egen wag = group(wave ageg gender)

* prep for ipfraking 
gen _one =1
svyset [pw = wgtr]
foreach x in rag dag wag {
	* summing weights for raking groups based on individuals who are in wave 15
	svy: total _one, over(`x', )
	matrix rake_`x' = e(b)
	matrix rowname rake_`x' = `x'
	
	*fix column names (ugh)
	quietly levelsof `x' if wgtr>0, local(levels)
	local coltext = ""
	foreach lvl in `levels' {
		local coltext = `"`coltext'"' + `""_one:`lvl'""'
	}
	mat colnames rake_`x' = `coltext'
}

* ipfraking on the three totals 
ipfraking if incovid==1 [pw = wgtr], ctotal(rake_rag rake_dag rake_wag) gen(wgtr_c) tol(0.01) nograph
foreach var in gender race age cendiv { 
	tab `var' [fw=wgtr]
	tab `var' if incovid==1 [fw=round(wgtr_c)]
} 

keep hhidpn wave wgtr_c 

* merge back 
tempfile covidwt 
save "`covidwt'"
frame change monthly 
merge m:1 hhidpn wave using "`covidwt'"
assert _merge==3 
drop _merge

* -------------------------------------------------------------------------- * 
* -------------------------------  add labels  ----------------------------- *
* -------------------------------------------------------------------------- * 
label define occ  1 "Management" ///
				  2 "Business+Financial Oper" ///
				  3 "Computer+Mathematical" ///
				  4 "Architecture+Engineering" ///
				  5 "Life/Physical/SocialSci" ///
				  6 "Community+Social Service" ///
				  7 "Legal" ///
				  8 "Education/Training/Lib" ///
				  9 "Arts/Design/Entertainment" ///
				 10 "Healthcare Practitioners" ///
				 11 "Healthcare Support" ///
				 12 "Protective Service" ///
				 13 "Food Prep+Serving Related" ///
				 14 "Building/Grounds Cleaning" ///
				 15 "Personal Care+Service" ///
				 16 "Sales+Related" ///
				 17 "Office+Admin Support" ///
				 18 "Farming/Fishing/Forestry" ///
				 19 "Construction+Extraction" ///
				 20 "Installation/Maintenance" ///
				 21 "Production" ///
				 22 "Transportation+Moving" ///
				 23 "Military Specific"
label values jcoccc occ 

label define ind 1  "Agric/Forest/Fish/Hunting"  ///
				 2  "Mining"  ///
				 3  "Utilities"  ///
				 4  "Construction"  ///
				 5  "Manufacturing"  ///
				 6  "Wholesale Trade"  ///
				 7  "Retail Trade"  ///
				 8  "Transport/Warehousing"  ///
				 9  "Information"  ///
				 10 "Finance/Insurance"  ///
				 11 "Real Estate/Rental/Leasing" /// 
				 12 "Prof/Scientific/Tech Svc"  ///
				 13 "Mgmnt/Admin/Support/Waste"  ///
				 14 "Educational Services"  ///
				 15 "Health Care/Social Asst"  ///
				 16 "Arts/Entertain/Recreation"  ///
				 17 "Accomodations/Food Svcs"  ///
				 18 "Other Svcs/Except PubAdm"  ///
				 19 "PubAdmin/Active Duty Mil" 
label values jcindc ind 

label define urbrur 1 "Urban" 2 "Suburban" 3 "Rural" 
label values urbrur urbrur 

label define cendiv 1 "New England" /// 
					2 "Mid Atlantic" /// 
					3 "EN Central" ///  
					4 "WN Central" /// 
					5 "S Atlantic" /// 
					6 "ES Central" /// 
					7 "WS Central" /// 
					8 "Mountain" /// 
					9 "Pacific" 
label values cendiv cendiv

label define shlt 1 "Excellent" ///
				  2 "Very good" ///
				  3 "Good" ///
				  4 "Fair" ///
				  5 "Poor"
label values shlt shlt 

* not sure why have to define these again but oh wel 
label define covwk_event 1 "LOST JOB/LAID OFF PERMANENTLY" ///
						 2 "FURLOUGHED/LAID OFF TEMPORARILY" ///
						 3 "QUIT" ///
						 4 "OTHER" ///
						 5 "Retired", replace  
label values covwk_event covwk_event

label define howaffect 1 "HAD TO CHANGE WORK DAYS OR HOURS" ///
					   2 "WORK BECAME MORE RISKY OR DANGEROUS" ///
					   3 "WORK BECAME HARDER" ///
					   4 "SWITCHED TO WORKING FROM HOME OR WORKING REMOTELY" ///
					   7 "OTHER" , replace  
label values covwk_howaffect howaffect

* finally finally finalize 
xtset hhidpn mo 
compress 

save hrs_monthly_temp, replace






