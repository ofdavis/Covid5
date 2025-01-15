* data setup 
clear all 
cd "/users/owen/Covid5"
global styear 2000

/* -----------------------------------------------------------------------------
									get IPUMS CPS data 
 -----------------------------------------------------------------------------
global asec 1
global bms 0
global start_yr = $styear
global start_mo = 3
global end_yr = 2024
global end_mo = 3
global var_list = "YEAR SERIAL MONTH  STATEFIP METRO PERNUM ASECFLAG CPSIDP AGE SEX RACE MARST HISPAN NATIVITY EMPSTAT LABFORCE OCC2010 IND1990 CLASSWKR WKSTAT ABSENT EDUC DIFFANY DIFFHEAR DIFFEYE DIFFREM DIFFPHYS DIFFMOB DIFFCARE VETSTAT SPLOC HRHHID FAMSIZE NCHILD ELDCH YNGCH WHYUNEMP DURUNEMP WORKLY IND90LY OCC10LY CLASSWLY WKSWORK1 WKSUNEM1 FULLPART PENSION FIRMSIZE WHYNWLY INCTOT INCSS INCRETIR INCDISAB INCDIVID INCRENT DISABWRK HEALTH QUITSICK PAIDGH HIMCAIDNW HIMCARENW OWNERSHP WHYSS1 WHYSS2"

do "/Applications/Stata/ado/personal/ipums_get.do"

assert asecflag==1
drop asecflag
save "data/asec_raw.dta", replace
*/

use data/asec_raw.dta, clear
cap drop hwtfinl  
format cpsidp %15.0f
gen mo = ym(year,month) 
format mo %tm

* covid dummy
gen covid = mo>=`=tm(2020m3)'


/* -----------------------------------------------------------------------------
									Backcast
This code implements the CPS weight backcasting described here: https://github.com/TheHamiltonProjectResearch/CPS-Population-Adjustment-Backcast-2012-22
The adjusted weights will be used for everything else 
-----------------------------------------------------------------------------*/

* Race/Ethnicity
gen ethnic=.
replace ethnic=6 if inrange(hispan,100,902) & ethnic==. 
replace ethnic=1 if race==100  & ethnic==. 
replace ethnic=2 if race==200  & ethnic==. 
replace ethnic=4 if race==651  & ethnic==. 
replace ethnic=5 if inlist(race, 300, 650) & ethnic==. 
replace ethnic=5 if inrange(race, 652, 830) & ethnic==. 
label define ethniclab 1 "white" 2 "Black" 4 "Asian" 5 "other" 6 "Hispanic" , replace
label values ethnic ethniclab

* Age Groups
gen agegr=.
replace agegr = 1 if inrange(age, 16, 17)
replace agegr = 2 if inrange(age, 18, 19) 
replace agegr = 3 if inrange(age, 20, 24) 
replace agegr = 4 if inrange(age, 25, 54) 
replace agegr = 5 if inrange(age, 35, 44) 
replace agegr = 6 if inrange(age, 45, 54) 
replace agegr = 7 if inrange(age, 55, 64) 
replace agegr = 8 if inrange(age, 65, 90) 

label define agegrlab 1 "16-17" 2 "18-19" 3 "20-24" 4 "25-34" 5 "35-44" 6 "45-54" 7 "55-64" 8 "65+"
label values agegr agegrlab

*merge backcast file in
merge m:1 month year sex agegr ethnic using data/cps_backcast, keepusing(ratio_adj) 
drop if _merge==2
drop _merge 

replace ratio_adj=1 if ratio_adj==.

replace asecwt = round(asecwt * ratio_adj)

drop ratio_adj agegr


/*----------------------------------------------------------------------------
								 Spouses
  ----------------------------------------------------------------------------*/
* Marital status 
gen married = 0
replace married = 1 if marst==1

* Create new frame to make and merge back in spouse data 
frame copy default spouse, replace
frame change spouse
	keep sploc marst serial cpsid hrhhid age year month
	keep if marst==1 & sploc!=0

	rename age age_sp
	rename sploc pernum
	rename cpsid cpsid_sp // for testing 

* merge back in
frame change default 
frame spouse: tempfile sp 
frame spouse: save "`sp'"
merge 1:1 serial pernum year month hrhhid using "`sp'", nogen
frame drop spouse

* Age groups, spouse. Note: 120 respondents 55+  have spouse age<18
	gen agegrp_sp = .
replace agegrp_sp = 55 if inrange(age_sp, 15, 59)
replace agegrp_sp = 60 if inrange(age_sp, 60, 61)
replace agegrp_sp = 62 if inrange(age_sp, 62, 64)
replace agegrp_sp = 65 if inrange(age_sp, 65, 69)
replace agegrp_sp = 70 if inrange(age_sp, 70, 85)

* for married respondents who have missing age_sp, make agegrp_sp same as agegrp
misstable sum agegrp_sp if married==1 
replace agegrp_sp = 55 if inrange(age, 15, 59) & married==1 & agegrp_sp==.
replace agegrp_sp = 60 if inrange(age, 60, 61) & married==1 & agegrp_sp==.
replace agegrp_sp = 62 if inrange(age, 62, 64) & married==1 & agegrp_sp==.
replace agegrp_sp = 65 if inrange(age, 65, 69) & married==1 & agegrp_sp==.
replace agegrp_sp = 70 if inrange(age, 70, 85) & married==1 & agegrp_sp==.

* create a separate category agegrp_sp=0 for these
replace agegrp_sp = 0 if married==0

* Drop younger observations -- no longer needed for spousal match 
drop if age<50

* family size 
gen child_any = nchild!=0
gen child_yng = yngch<=18
gen child_adt = eldch>=18 & eldch<99

tab yngch if nchild>1


/*----------------------------------------------------------------------------
								 Demographic vars 
  ----------------------------------------------------------------------------*/
replace sex = sex-1
label define sex 0 "man" 1 "woman"
label values sex sex 

* age: make consistent (2002-2004 will not have any 85+)
replace age = 85 if age>=85
replace age = 80 if inrange(age,80,84)

gen agesq = age^2
gen agecub = age^3

* educ
gen educ_ = 0 
replace educ_ = 1 if educ==73
replace educ_ = 2 if inrange(educ,81,110)
replace educ_ = 3 if educ==111
replace educ_ = 4 if inrange(educ,112,125)

label define educ 0 "less than hs" 1 "hs" 2 "some college" 3 "bachelor" 4 "advanced"
label values educ_ educ

drop educ
rename educ_ educ

* race
drop race 
rename ethnic race

* native 
tab nativity
recode nativity (0=3) (1=0) (2/4=1) (5=2)

* metro area
rename metro metro_cps
gen metro = inrange(metro_cps, 2, 4)

* veteran 
tab vetstat
replace vetstat=0 if vetstat==1
replace vetstat=1 if vetstat==2
rename vetstat vet
label drop vetstat_lbl 

* disability 
gen disable = 0 
replace disable=1 if diffany==2

replace diffrem = diffrem==2
replace diffphys = diffphys==2
replace diffmob = diffmob==2

* own house 
gen own = ownershp==10

* some labeling
label define covid_lbl 0 " " 1 "Covid"
label values covid covid_lbl

label var mo "date"

drop *_cps 



/*----------------------------------------------------------------------------
								Employment vars 
  ----------------------------------------------------------------------------*/
*employment status
gen emp=.
replace emp=1 if empstat==10 | empstat==12 // employed
replace emp=2 if empstat==21 | empstat==22 // unemployed
replace emp=3 if empstat==32 | empstat==34 					// not in LF other
replace emp=4 if empstat==36 // not in LF ret
drop if emp==. // armed forces

*retired status, employed status
gen retired = empstat==36
gen employed = emp==1
gen unem = emp==2
gen nlf = emp==3
gen unable = empstat==32
gen nlf_oth = empstat==34 
gen untemp = whyunemp==1
gen unlose = whyunemp==2

* self employed
gen self = inrange(classwkr, 10, 14)
gen selfly = inrange(classwly, 10, 14)

* public vs private
gen govt = inrange(classwkr, 24, 28)
gen govtly = inrange(classwly, 24, 28)

* full time worker
gen ft = (wkstat>=10 & wkstat<=15) | wkstat==50

* absent 
gen absnt = absent==3 

* duration unemployment -- impute if dur==999 and unem==1 (doesn't happen in CPS, just 70 cases)
gen dur = durunemp if unem==1 
gen dur_flag = dur==999 & unem==1
reg durunem i.sex c.age##c.age i.educ i.race i.empstat year if durunem<999
predict p_dur
sum dur p_dur if unem==1
replace dur = p_dur if dur_flag==1
drop p_dur dur_flag



/*----------------------------------------------------------------------------
								ASEC vars 
  ---------------------------------------------------------------------------- */
* workly 
recode workly (1=0) (2=1)
label define workly 0 "no" 1 "yes"
label values workly workly

* other ly work vars 
rename wkswork1 wksly 
gen wksly_lt52 = wksly<52
gen unemly = inrange(wksunem1,1,51)

* disability affecting work
recode disabwrk (1=0) (2=1)
label define disabwrk 0 "no" 1 "yes"
label values disabwrk disabwrk

* total income quintiles 
gen incq = . 
qui levelsof year, local(years)
foreach y of local years {
	di `y'
	xtile incq_=inctot if year==`y' [fw=asecwt], n(4)
	replace incq=incq_ if year==`y' 
	drop incq_
}

* indicate if ss income (of at least 1st percentile )
gen ssinc=0
qui levelsof year, local(years)
foreach y of local years {
	di `y'
	sum incss if year==`y' & incss>0 [fw=asecwt], d
	replace ssinc=1 if year==`y' & incss>r(p1)
}

* indicate if non-ss retirement income (of at least 1st percentile )
gen retinc=0
qui levelsof year, local(years)
foreach y of local years {
	di `y'
	sum incretir if year==`y' & incretir>0 & incretir<99999999 [fw=asecwt], d
	replace retinc=1 if year==`y' & incretir>r(p1) & incretir<99999999
}

* indicate if divid income (of at least 1st percentile )
gen divinc=0
qui levelsof year, local(years)
foreach y of local years {
	di `y'
	sum incdivid if year==`y' & incdivid>0 & incdivid<999999 [fw=asecwt], d
	replace divinc=1 if year==`y' & incdivid>r(p1) & incdivid<999999
}

* indicate if rental income (of at least 1st percentile )
gen rentinc=0
qui levelsof year, local(years)
foreach y of local years {
	di `y'
	sum incrent if year==`y' & incrent>0 & incrent<9999999 [fw=asecwt], d
	replace rentinc=1 if year==`y' & incrent>r(p1) & incrent<9999999
}

* inc from rent or divid 
gen incrd = rentinc==1 | divinc==1

* if ss-retirement 
gen ssret = whyss1==1



* to ignore for now: himcaidnw himcarenw pension firmsize paidgh quitsick incdisab 

/*----------------------------------------------------------------------------
							industry and occupation 
  ---------------------------------------------------------------------------- */
*maj industry grps
forvalues ly=0/1 { 
	if `ly'==0 local s ""
	if `ly'==0 local y "1990"
	if `ly'==1 local s "ly"
	if `ly'==1 local y "90"
	gen     ind_maj`s' = .
	replace ind_maj`s' = 1 if  inrange(ind`y'`s', 10, 32)    	// ag etc
	replace ind_maj`s' = 2 if  inrange(ind`y'`s', 40, 50)    	// mining
	replace ind_maj`s' = 3 if  inrange(ind`y'`s', 60, 60)    	// construction
	replace ind_maj`s' = 4 if  inrange(ind`y'`s', 100, 392)   	// manuf
	replace ind_maj`s' = 5 if  inrange(ind`y'`s', 400, 472) 		// trans/util
	replace ind_maj`s' = 6 if  inrange(ind`y'`s', 500, 571)   	// wholesale
	replace ind_maj`s' = 7 if  inrange(ind`y'`s', 580, 691)   	// retail
	replace ind_maj`s' = 8 if  inrange(ind`y'`s', 700, 712)   	// financial
	replace ind_maj`s' = 9 if  inrange(ind`y'`s', 721, 760)   	// biz and repair serv
	replace ind_maj`s' = 10 if inrange(ind`y'`s', 761, 791)   	// personal serv
	replace ind_maj`s' = 11 if inrange(ind`y'`s', 800, 810)   	// entertainment and rec
	replace ind_maj`s' = 12 if inrange(ind`y'`s', 812, 893)   	// prof and related
	replace ind_maj`s' = 13 if inrange(ind`y'`s', 900, 932)   	// public admin
	replace ind_maj`s' = 14 if inrange(ind`y'`s', 940, 998)   	// military
}

label define ind_maj_lbl 1 "Agriculture and related" ///
						 2 "Mining, quarrying, and oil and gas extraction" ///
						 3 "Construction" ///
						 4 "Manufacturing" ///
						 5 "Transportation and utilities" ///
						 6 "Wholesale trade" ///
						 7 "Retail trade" /// 
						 8 "Financial activities" ///
						 9 "Business and repair services" ///
						 10 "Personal services" ///
						 11 "Entertainment and recreation" ///
						 12 "Professional and related" ///
						 13 "Public administration"  ///
						 14 "Military" 

label values ind_maj ind_maj_lbl 
label values ind_majly ind_maj_lbl 

* major occ groups 
forvalues ly=0/1 { 
	if `ly'==0 local s ""
	if `ly'==0 local y "2010"
	if `ly'==1 local s "ly"
	if `ly'==1 local y "10"
		gen occ_maj`s'=.
	replace occ_maj`s'=1	if	inrange(occ`y'`s', 10  , 440 )
	replace occ_maj`s'=2	if	inrange(occ`y'`s', 500 , 960 )
	replace occ_maj`s'=3	if	inrange(occ`y'`s', 1000, 1240)
	replace occ_maj`s'=4	if	inrange(occ`y'`s', 1300, 1560)
	replace occ_maj`s'=5	if	inrange(occ`y'`s', 1600, 1980)
	replace occ_maj`s'=6	if	inrange(occ`y'`s', 2000, 2060)
	replace occ_maj`s'=7	if	inrange(occ`y'`s', 2100, 2180)
	replace occ_maj`s'=8	if	inrange(occ`y'`s', 2200, 2555)
	replace occ_maj`s'=9	if	inrange(occ`y'`s', 2600, 2960)
	replace occ_maj`s'=10	if	inrange(occ`y'`s', 3000, 3550)
	replace occ_maj`s'=11	if	inrange(occ`y'`s', 3600, 3655)
	replace occ_maj`s'=12	if	inrange(occ`y'`s', 3700, 3960)
	replace occ_maj`s'=13	if	inrange(occ`y'`s', 4000, 4160)
	replace occ_maj`s'=14	if	inrange(occ`y'`s', 4200, 4255)
	replace occ_maj`s'=15	if	inrange(occ`y'`s', 4300, 4655)
	replace occ_maj`s'=16	if	inrange(occ`y'`s', 4700, 4965)
	replace occ_maj`s'=17	if	inrange(occ`y'`s', 5000, 5940)
	replace occ_maj`s'=18	if	inrange(occ`y'`s', 6005, 6130)
	replace occ_maj`s'=19	if	inrange(occ`y'`s', 6200, 6950)
	replace occ_maj`s'=20	if	inrange(occ`y'`s', 7000, 7640)
	replace occ_maj`s'=21	if	inrange(occ`y'`s', 7700, 8990)
	replace occ_maj`s'=22	if	inrange(occ`y'`s', 9000, 9760)
	replace occ_maj`s'=23	if	inrange(occ`y'`s', 9830, 9999)
}

// value labels
label define occ_maj_lbl 1 "Management " ///
						 2 "Business and Financial Operations " ///
						 3 "Computer and Mathematical " ///
						 4 "Architecture and Engineering " ///
						 5 "Life, Physical, and Social Science " ///
						 6 "Community and Social Services " ///
						 7 "Legal " ///
						 8 "Education, Training, and Library " ///
						 9 "Arts, Design, Entertainment, Sports, and Media " ///
						 10 "Healthcare Practitioners and Technical " ///
						 11 "Healthcare Support " ///
						 12 "Protective Service " ///
						 13 "Food Preparation and Serving Related " ///
						 14 "Building and Grounds Cleaning and Maintenance " ///
						 15 "Personal Care and Service " ///
						 16 "Sales and Related " ///
						 17 "Office and Administrative Support " ///
						 18 "Farming, Fishing, and Forestry " ///
						 19 "Construction Trades and Extraction Workers" ///
						 20 "Installation, Maintenance, and Repair Workers" ///
						 21 "Production " ///
						 22 "Transportation and Material Moving " ///
						 23 "Armed Forces"
label values occ_maj occ_maj_lbl
label values occ_majly occ_maj_lbl

rename occ2010 occ 
rename ind1990 ind

rename occ10ly occly 
rename ind90ly indly


/*----------------------------------------------------------------------------
								Social security PIA 
  ----------------------------------------------------------------------------*/
* 1. For each age/year combo, provide 12 possible birth dates (year and month, assume born on last day of month)
forvalues m=1/12 { 
	gen bdate`m' = mo-((age+1)*12)+`m'-1
	format bdate`m' %tm
}

/* need a new approach here that avoids missings 
-- for each bmonth, calculate all possible PIA values  
-- rescale so that max = 1 
-- for any age<62, replace pia with pia at 62m1 
-- for any age>70, replace pia = 1 
-- create dummy indicating age in [62,69]
*/ 

frame2 pia, replace 
set obs 732 // enough for bdates between 1924-1975 
gen bdate = _n -1 + `=tm(1914m1)'
format bdate %tm 

* 2. For each birth date, calculate time until/since full retirement age
gen rdate = bdate ///
	+ (year(dofm(bdate))<=1937) * (65*12 + 0 ) ///
	+ (year(dofm(bdate))==1938) * (65*12 + 2 ) /// 
	+ (year(dofm(bdate))==1939) * (65*12 + 4 ) /// 
	+ (year(dofm(bdate))==1940) * (65*12 + 6 ) /// 
	+ (year(dofm(bdate))==1941) * (65*12 + 8 ) /// 
	+ (year(dofm(bdate))==1942) * (65*12 + 10) ///
	+ (year(dofm(bdate))==1943) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1944) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1945) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1946) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1947) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1948) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1949) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1950) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1951) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1952) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1953) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1954) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1955) * (66*12 + 2 ) /// 
	+ (year(dofm(bdate))==1956) * (66*12 + 4 ) /// 
	+ (year(dofm(bdate))==1957) * (66*12 + 6 ) /// 
	+ (year(dofm(bdate))==1958) * (66*12 + 8 ) /// 
	+ (year(dofm(bdate))==1959) * (66*12 + 10) ///
	+ (year(dofm(bdate))>=1960) * (67*12 + 0 )
format rdate %tm


* 3. For each possible birth date, calculate PIA 
* calculate gains 
gen gain = .
replace gain = 0.03  if inrange(year(dofm(bdate)),1914, 1924) 
replace gain = 0.035 if inrange(year(dofm(bdate)),1925, 1926) 
replace gain = 0.04  if inrange(year(dofm(bdate)),1927, 1928) 
replace gain = 0.045 if inrange(year(dofm(bdate)),1929, 1930) 
replace gain = 0.05  if inrange(year(dofm(bdate)),1931, 1932) 
replace gain = 0.055 if inrange(year(dofm(bdate)),1933, 1934) 
replace gain = 0.06  if inrange(year(dofm(bdate)),1935, 1936) 
replace gain = 0.065 if inrange(year(dofm(bdate)),1937, 1938) 
replace gain = 0.07  if inrange(year(dofm(bdate)),1939, 1940) 
replace gain = 0.075 if inrange(year(dofm(bdate)),1941, 1942) 
replace gain = 0.08  if year(dofm(bdate))>=1943

* expand to length of sample -- long enough to get FRA for every possible bdate 
qui sum rdate 
scalar min = r(min)
scalar max = r(max)
local span = max + 36 - min + 1 
di `span'
expand `span'
bys bdate: gen mo = _n
replace mo = mo - 1 + min 
format mo %tm

gen age = floor((mo-bdate)/12)

* calculate PIA 
gen pia = . 
replace pia = (-5/9)*0.01*(rdate-mo) if inrange(rdate-mo,1,36) & age>=62 & mo<rdate
replace pia = (-5/9)*0.01*36 + (-5/12)*0.01*(rdate-mo-36) if (rdate-mo)>36 & age>=62 & mo<rdate
replace pia = gain*(mo-rdate)/12 if mo>=rdate & ///
	(age<70 | (age==70 & month(dofm(bdate))==month(dofm(mo)))) 
	
* pia as factor 
gen pia1 = pia + 1 
bys bdate (mo): egen pia1_max = max(pia1)
bys bdate (mo): egen pia1_min = min(pia1)
replace pia1=pia1_max if age>=70
replace pia1=pia1_min if age<62

keep bdate mo pia1
rename pia1 pia 

* merge to default frame each possible bdate 
frame change default 
forvalues m=1/12 { 
	frame pia: rename bdate* bdate`m'
	frame pia: rename pia* pia`m'
	frame pia: tempfile pia 
	frame pia: save "`pia'"
	merge m:1 mo bdate`m' using "`pia'", nogen keep(match master)
}
*br year month age bdate* pia* 

egen pia = rowmean(pia1 pia2 pia3 pia4 pia5 pia6 pia7 pia8 pia9 pia10 pia11 pia12) 

drop bdate1-bdate12 
drop pia1-pia12 

gen ssa = inrange(age,62,69)


/*----------------------------------------------------------------------------
								urates 
  ----------------------------------------------------------------------------*/
frame2 urate, replace 
import fred UNRATE, daterange(${styear}-01-01 2024-08-01) aggregate(monthly) 
gen mo = ym(year(daten),month(daten))
format mo %tm
drop date* 
rename UNRATE ur 

* CBO urate 
frame2 urate2, replace 
import fred NROU, daterange(${styear}-01-01 2024-08-01) aggregate(quarterly) 
expand 3
sort daten
gen month = month(daten)
gen year = year(daten)
egen add = fill(0 1 2 0 1 2)
gen month1 = month+add
gen mo = ym(year,month1)
format mo %tm
drop date* month* year
rename NROU cbo 

frame change urate 
frame urate2: tempfile urate2 
frame urate2: save "`urate2'"
merge 1:1 mo using "`urate2'"
assert _merge!=1
drop if _merge==2
drop _merge

gen urhat = ur-cbo 
sum urhat if inrange(mo,`=tm(2019m10)', `=tm(2019m12)')
replace urhat = r(mean) if mo>=`=tm(2020m1)'
keep mo ur urhat

frame change default 
frame urate: tempfile urate 
frame urate: save "`urate'"
merge m:1 mo using "`urate'"
assert _merge!=1
drop if _merge==2
drop _merge

assert ur!=.


/*----------------------------------------------------------------------------
								finalize and save 
  ----------------------------------------------------------------------------*/
compress

keep year mo covid statefip asecwt cpsidp age sex ///
	 vet diffrem diffphys diffmob race nativity ///
	 famsize child_any child_yng child_adt agegrp_sp married ///
	 agesq agecub educ metro emp employed retired unem nlf ///
	 dur untemp unlose unable nlf_oth pia ur urhat ssa ///
	 own workly wksly wksly_lt52 unemly fullpart selfly govtly ind_majly occ_majly ///
	 whynwly health incq ssinc retinc ssret incrd 

save data/asec_data.dta, replace




