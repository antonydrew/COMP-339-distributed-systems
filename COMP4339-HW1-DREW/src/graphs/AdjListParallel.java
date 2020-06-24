package graphs;

import java.util.*;
import graphs.AdjList;


public class AdjListParallel implements Runnable
{
   private static int Out;
   private static int tupleBegin;
   private static int tupleEnd;
   private static AdjList adjacList;
   private static AdjList metaList;
   
   public AdjListParallel(int Out, int tupleBegin, int tupleEnd, AdjList adjacList, AdjList metaList)
   {
        
		AdjListParallel.Out = Out;
		AdjListParallel.tupleBegin = tupleBegin;
		AdjListParallel.tupleEnd = tupleEnd;
		AdjListParallel.adjacList = adjacList;
		AdjListParallel.metaList = metaList;
		
   }

   public synchronized static int getOut() { 
		System.out.println("Local Thread Sum of Weights is: " + AdjListParallel.Out);
		 
		return AdjListParallel.Out;
	}
   
   public synchronized void setOut(int Out) { 
		AdjListParallel.Out = Out;
	}
     
   public void run() {
	   setOut(getWeightsParallel(adjacList, metaList, tupleBegin, tupleEnd));
	}
  
   public static int getWeightsParallel(AdjList adjacList1, AdjList metaList1, int tupleBegin, int tupleEnd) 
	{     
	  int score = 0; int mlen = 0, elen = 0; int totalWeight = 0;
	  double floor = 0.00; 
	
	  for (int k = tupleBegin; k < tupleEnd; k++)
      {		   
	  List<String> mList = metaList.getTuple(Integer.toString(k));
      List<String> eList = adjacList.getTuple(Integer.toString(k));
      mlen = mList.size();
      elen = eList.size();
   
      int [] localRanks = new int[elen];
      score = 0; 
      
      if (mlen >= 4 && elen > 0) //skip discontinued products and no-edge products
      {
      for (int i = 0; i < elen; i++)
       {
 	   
      		List<String> mList1 = metaList.getTuple(eList.get(i));
      		try 
      		{      			
      			localRanks[i] = Integer.parseInt(mList1.get(3));
      		}
      	    catch (Exception e) { localRanks[i] = 99999999; }  //discontinued sub-product missing rank so assign INF	
      		if (Integer.parseInt(mList.get(3)) <= localRanks[i]) score++;   		
       } 
      }
      else score = 0;
      
      floor = Math.floor(elen/2);
      if (score >= floor && elen > 0) metaList.addMeta(Integer.toString(k), "1");
      else metaList.addMeta(Integer.toString(k), "0");
      
      //check for extra weight at end of metaTable if ran SEQ weights first
      if (mlen >= 5) totalWeight = totalWeight + Integer.parseInt(mList.get(5)); 
      else if (mlen >= 4) totalWeight = totalWeight + Integer.parseInt(mList.get(4));    
     
      }
	
      return totalWeight;   	   
	}     
 
 
}

