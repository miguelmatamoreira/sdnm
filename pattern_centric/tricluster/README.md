# tricluster

1. ``make``

2. ``./triCluster -fFile -s[Times,Samples,Genes] [other options]``

        File Name:           -fString
        Minimum Size:        -s[Integer,Integer,Integer]   /*T, S, G*/
        Range Window Size:   -wFloat   /*0.01 by default*/
        Deletion Threshold:  -dFloat   /*1.00 by default*/
        Merging  Threshold:  -mFloat   /*1.00 by default*/
        Unrelated Numbers:   -uFloat   /*mciroCluster will not consider the values refered by this option*/
        delta-T:             -etFloat  /*when trying to get close Time values only*/
        delta-S:             -esFloat  /*when trying to get close Sample values only*/
        delta-G:             -egFloat  /*when trying to get close Gene values only*/
        Record to File:      -rString  /*Output the result to a file */
        Output in Brief:     -b        /*Output the current status in brief*/
        Output Clusters:     -o123     /*1:Original clusters,  2:Clusters  after deletion, 3:Clusters  after merging*/
        Output Names:        -ntsg     /*t:time names, s:Sample names, g:Gene names*/
