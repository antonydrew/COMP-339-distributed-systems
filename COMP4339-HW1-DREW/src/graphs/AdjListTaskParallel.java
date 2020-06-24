package graphs;

import java.util.*;
import graphs.AdjListTask;


public class AdjListTaskParallel implements Runnable
{
   private static int Out;
   private static int tupleBegin;
   private static int tupleEnd;
   private static String src;
   private static AdjListTask adjacList;
   
   public AdjListTaskParallel(int Out, int tupleBegin, int tupleEnd, String src, AdjListTask adjacList)
   {
        
		AdjListTaskParallel.Out = Out;
		AdjListTaskParallel.tupleBegin = tupleBegin;
		AdjListTaskParallel.tupleEnd = tupleEnd;
		AdjListTaskParallel.src = src;
		AdjListTaskParallel.adjacList = adjacList;
		
   }

   public synchronized static int getOut() { 
	   System.out.println("Local Thread Sum of vertices is: " + AdjListTaskParallel.Out);
		return AdjListTaskParallel.Out;
	}
   
   public synchronized void setOut(int Out) { 
		AdjListTaskParallel.Out = Out;
	}
     
   public void run() {
	   setOut(getWeightsParallel(adjacList, src));
	}
  
   public static int getWeightsParallel(AdjListTask adjacList, String src) 
	{     
	  int totalWeight = 0; int tupleEnd = Integer.parseInt(src);
	
	  for (int k = 0; k < tupleEnd; k++)
      {		   

      totalWeight = totalWeight + k;     
     
      }
	
      return totalWeight;   	   
	}     
 
}

