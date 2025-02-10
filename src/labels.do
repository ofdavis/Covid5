qui { 
/* global label generation utility
foreach var in married sex vet nativity race  agegrp_sp child_any child_yng child_adt educ metro unable nlf_oth untemp unlose self govt ft absnt ind_maj occ_maj { 
	local lblname: value label `var'  // Get value label name associated with `var`
    

	qui levelsof `var', local(values)  
	di ""
	di "* `var'"
	foreach v of local values { 
		
		if "`lblname'"=="" {
			di `"global `var'`v' = "" "' 
		}
		else {
			local lbl: label `lblname' `v' 
			di `"global `var'`v' = "`lbl'""' 
		}
	}
}
*/ 


* married
global married_lab "Marital status"
global married0 = "Unmarried" 
global married1 = "Married" 

* sex
global sex_lab "Gender"
global sex0 = "Men"
global sex1 = "Women"

* vet
global vet_lab "Veteran status"
global vet0 = "not a veteran"
global vet1 = "veteran"

* nativity
global nativity_lab "Nativity"
global nativity0 = "Native-born, both parents"
global nativity1 = "Native-born, foreign parent(s)"
global nativity2 = "Foreign-born"
global nativity3 = "Unknown"

* foreign-born
global foreign_lab "Nationality"
global foreign0 "Native-born"
global foreign1 "Foreign-born"

* race
global race_lab "Race/ethnicity"
global race1 = "White"
global race2 = "Black"
global race3 = "Hispanic" // for relabeled race (twoway bar)
global race4 = "Asian"
global race5 = "Other"
global race6 = "Hispanic"

* agegrp_sp
global agegrp_sp_lab "Age group of spouse"
global agegrp_sp0 = "No spouse" 
global agegrp_sp55 = "Spouse 59 or younger" 
global agegrp_sp60 = "Spouse 60-61" 
global agegrp_sp62 = "Spouse 62-64" 
global agegrp_sp65 = "Spouse 65-69" 
global agegrp_sp70 = "Spouse 70 and older " 

* child_any
global child_any_lab "Any own children in household"
global child_any0 = "None" 
global child_any1 = "Own children in the house" 

* child_yng
global child_yng_lab "Own minor child in household"
global child_yng0 = "None" 
global child_yng1 = "Own minor child in the house" 

* child_adt
global child_adt_lab "Own adult child in household"
global child_adt0 = "None" 
global child_adt1 = "Own adult child in the house" 

* educ
global educ_lab "Education"
global educ0 = "Less than high school"
global educ1 = "High school"
global educ2 = "Some college"
global educ3 = "Bachelor"
global educ4 = "Advanced"

* metro
global metro_lab "Metro status"
global metro0 = "Non-metro area" 
global metro1 = "Metro area" 

* unable
global unable_lab "Whether NLF because unable to work"
global unable0 = "Not unable to work" 
global unable1 = "NLF, unable to work" 

* nlf_oth
global nlf_oth_lab "Whether NLF for other reason"
global nlf_oth0 = "Not NLF, other reason" 
global nlf_oth1 = "NLF, other reason" 

* untemp
global untemp_lab "Whether temporarily unemployed"
global untemp0 = "Not temporarily unemployed" 
global untemp1 = "Temporarily unemployed" 

* unlose
global unlose_lab "Whether permanent job loser"
global unlose0 = "No permanent job loser" 
global unlose1 = "Permanent job loser" 

* self
global self_lab "Whether self-employed"
global self0 = "Not self-employed" 
global self1 = "Self-employed" 

* govt
global govt_lab "Whether public-sector employee"
global govt0 = "Not a public employee" 
global govt1 = "Public employee" 

* ft
global ft_lab "Work schedule"
global ft0 = "Part-time" 
global ft1 = "Full-time" 

* absnt
global absnt_lab "Whether absent from work"
global absnt0 = "Not absent from work" 
global absnt1 = "Absent from work" 

* ind_maj
global ind_maj_lab "Major industry of employment"
global ind_maj1  = "Agriculture and related" 
global ind_maj2  = "Mining" 
global ind_maj3  = "Construction" 
global ind_maj4  = "Manufacturing"
global ind_maj5  = "Transportation and utilities" 
global ind_maj6  = "Wholesale trade" 
global ind_maj7  = "Retail trade" 
global ind_maj8  = "Financial activities" 
global ind_maj9  = "Business and repair services"
global ind_maj10 = "Personal services" 
global ind_maj11 = "Entertainment and recreational services" 
global ind_maj12 = "Professional and related services" 
global ind_maj13 = "Public administration"  
global ind_maj14 = "Military" 

* occ_maj
global occ_maj_lab "Major occupation of employment "
global occ_maj1 = "Management"
global occ_maj2 = "Business and Financial Operations"
global occ_maj3 = "Computer and Mathematical"
global occ_maj4 = "Architecture and Engineering"
global occ_maj5 = "Life, Physical, and Social Science"
global occ_maj6 = "Community and Social Services"
global occ_maj7 = "Legal"
global occ_maj8 = "Education, Training, and Library"
global occ_maj9 = "Arts, Design, Entertainment, Sports, and Media"
global occ_maj10 = "Healthcare Practitioners and Technical"
global occ_maj11 = "Healthcare Support"
global occ_maj12 = "Protective Service"
global occ_maj13 = "Food Preparation and Serving Related"
global occ_maj14 = "Building and Grounds Cleaning and Maintenance"
global occ_maj15 = "Personal Care and Service"
global occ_maj16 = "Sales and Related"
global occ_maj17 = "Office and Administrative Support"
global occ_maj18 = "Farming, Fishing, and Forestry"
global occ_maj19 = "Construction Trades and Extraction Workers"
global occ_maj20 = "Installation, Maintenance, and Repair Workers"
global occ_maj21 = "Production"
global occ_maj22 = "Transportation and Material Moving"
global occ_maj23 = "Armed Forces"

* birth cohort 
global cohort_lab "Birth cohort"
global cohort1 "50-59 in 2020" 
global cohort2 "60-64 in 2020"
global cohort3 "65-69 in 2020"
global cohort4 "70-74 in 2020"
global cohort5 "75+ in 2020"

* for graphing
global covid_line xline(`=tm(2020m4)', lc(black%25) lp(dash))
global covid_line_yr xline(2020, lc(black%25) lp(dash))

}
