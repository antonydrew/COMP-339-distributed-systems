package graphs;

import java.io.*;
import java.util.*;
import graphs.AdjList;


public class AdjListParallelIO implements Runnable
{
   
   private static int tupleBegin;
   private static int tupleEnd;  
   private static String fileName;	
   private static AdjList graph;
   
   public AdjListParallelIO(int tupleBegin, int tupleEnd, String fileName, AdjList graph)
   {
        	
		AdjListParallelIO.tupleBegin = tupleBegin;
		AdjListParallelIO.tupleEnd = tupleEnd;	
		AdjListParallelIO.fileName = fileName;
		AdjListParallelIO.graph = graph;
   }
  
   public void run() {
	   createGraphParallel(fileName, tupleBegin, tupleEnd, graph);
	}
  
   public synchronized static void createGraphParallel(String fileName, int tupleBegin, int tupleEnd, AdjList graph) {
	   
	   int counter = 1; String src, desty; int gap = 4;
	   if (gap > tupleBegin) gap = 4;
	   else gap = 0;
	   FileReader file = null;
	   BufferedReader reader = null;
	   String[] values = new String [10];	 
	   
	   try 
	   {
	     file = new FileReader(fileName);
	     reader = new BufferedReader(file);
	     
	     if (counter < tupleBegin) {
	     for (int j =1; j < tupleBegin; j++) {
	    	 reader.readLine(); 
	    	 counter++;} //skip over rows at beginning if tupleBegin does NOT equal ZERO
	     }
	     
	     String line = "";
	     while ((line = reader.readLine()) != null){
	          values= line.split("\t");
	          if (values[0].equals(Integer.toString(tupleEnd))) break;
	          if (counter > gap && counter >= tupleBegin+gap) {
	    	   src = values[0]; desty = values[1];
	    	   List<String>check = graph.getTuple(src);
	    	   if (!check.contains(desty)) graph.addEdge(src,desty);
	       }
	         counter++;    	 
	     }
	   } catch (Exception e) {
		   System.out.println("Num of Vertices is too small for data - Parallel graph error! "
		   		+ "try running again with larger input!\n");
	   } finally {
	     if (file != null) {
	       try {
	         file.close();
	         reader.close();
	       } catch (IOException e) {
	       }
	     }
	   }
	  
	 } 
 
}

