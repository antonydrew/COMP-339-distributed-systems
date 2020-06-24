package graphs;

import java.io.*;
import java.util.*;
import graphs.AdjListParallel;
import graphs.AdjListParallelIO;

import static java.lang.Thread.sleep;


public class AdjList extends LinkedHashMap<String, LinkedList<String>>
{
   
   private static final long serialVersionUID = 1L;
   private  Map<String,LinkedList<String>> AdList;	
   
   public AdjList(int numVertices)
   {
	 	
	    AdList = (new LinkedHashMap<String, LinkedList<String>>());		
        for (int i = 0; i < numVertices;  i++)
        {
           AdList.put(Integer.toString(i), new LinkedList<String>());
        }
   }
  
   public void addMeta(String pID, String meta)
   {
	   
       List<String> srcMetaList = AdList.get(pID);
       srcMetaList.add(meta);
      
   }
   
   public void addEdge(String src, String desty)
   {

       List<String> srcList = AdList.get(src);
       srcList.add(desty);
      
   }

   public LinkedList<String> getTuple(String src)
   {
       if (Integer.parseInt(src) > AdList.size())
       {
           System.out.println("Vertex not in graph");
            return null;
        }

        return AdList.get(src);
    }
   
   public static int getWeights(int numVertices, AdjList adjacList, AdjList metaList) 
	{     
	   int totalWeight = 0; int score = 0; int mlen = 0, elen = 0;
	   double floor = 0.00;
	   
	   for (int k = 0; k < numVertices; k++)
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
       	    catch (Exception e) { localRanks[i] = 99999999;} //discontinued sub-product missing rank so assign INF
       		if (Integer.parseInt(mList.get(3)) <= localRanks[i]) score++;    		
        } 
       }
       else score = 0;
       
       floor = Math.floor(elen/2);
       if (score >= floor && elen > 0) metaList.addMeta(Integer.toString(k), "1");
       else metaList.addMeta(Integer.toString(k), "0");

       //check for extra weight at end of metaTable if ran PARA weights first
       if (mlen >= 5) totalWeight = totalWeight + Integer.parseInt(mList.get(5)); 
       else if (mlen >= 4) totalWeight = totalWeight + Integer.parseInt(mList.get(4));    
       }
	   
       return totalWeight;   	   
	}     
   
   public static int getNumVertices(String fileName) {
	   
	   int counter = 1;
	   int numVertices = 0;
	   FileReader file = null;
	   BufferedReader reader = null;
	   String[] values = new String [10];	   
	   
	   try 
	   {
	     file = new FileReader(fileName);
	     reader = new BufferedReader(file);
	     String line = "";
	     while ((line = reader.readLine()) != null && counter < 5) {
	       values= line.split(" ");
	       if (counter == 3 && values[1].equals("Nodes:")) {
	    	   numVertices = Integer.parseInt(values[2]);
	       }
	       counter++;
	     }
	   } catch (Exception e) {
		   System.out.println("No file or data found!");
	   } finally {
	     if (file != null) {
	       try {
	         file.close();
	         reader.close();
	       } catch (IOException e) {
	       }
	     }
	   }
	   
	   return numVertices;
	 } 
     
   public static AdjList createGraph(String fileName, int numVertices) {
	   
	   int counter = 1; String src, desty; 
	   AdjList adjacList = new AdjList(numVertices);
	   FileReader file = null;
	   BufferedReader reader = null;
	   String[] values = new String [numVertices];	   
	   
	   try 
	   {
	     file = new FileReader(fileName);
	     reader = new BufferedReader(file);
	     String line = "";
	     while ((line = reader.readLine()) != null) {
	       values= line.split("\t");
	       if (counter > 4) {
	    	   src = values[0]; desty = values[1];
	    	   adjacList.addEdge(src,desty);
	       }
	       counter++;
	     }
	   } catch (Exception e) {
		   System.out.println("Num of Vertices is too small for data - Seq graph error! "
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
	   
	   return adjacList;
	 } 
    
   public static AdjList createMetaDat(String fileName, int numVertices) {
	   
	   int counter = 1, gap=0, lens=0; String pID = "", sRank="", title="", group="";
	   AdjList metaList = new AdjList(numVertices);
	   FileReader file = null;
	   BufferedReader reader = null;
	   String[] values = new String [100];
	   
	   try 
	   {
	     file = new FileReader(fileName);
	     reader = new BufferedReader(file);
	     String line = "";
	     while ((line = reader.readLine()) != null) {
	       if (line.trim().length() == 0) continue;
	       line = line.trim(); 
	       values= line.split(" ");
	       
	       if (counter > 2 && values[0].equals("Id:") && values[3].equals(Integer.toString(numVertices))) break; //skip rest
	       if (counter > 2 && values[0].equals("Id:") && values[3]!=null) {
	    	   pID = values[3]; gap=0; metaList.addMeta(pID, pID);}
	       if (counter > 2 && values[0].equals("title:") && values[1]!=null && gap==0) {
	    	   lens = values.length;
	    	   for (int i =1; i < lens; i++) {
	    		   title = title+values[i];
	    		   if (i<lens-1) title = title + " ";
	    	   }
	    	   gap++; metaList.addMeta(pID, title);
	    	   title=""; }
	       if (counter > 2 && values[0].equals("group:") && values[1]!=null && gap==1) {
	    	   group = values[1]; gap++; metaList.addMeta(pID, group);}
	       if (counter > 2 && values[0].equals("salesrank:") && values[1]!=null && gap==2) {
	    	   sRank = values[1]; gap++; metaList.addMeta(pID, sRank);}
	       counter++;
	     }
	   } catch (Exception e) {
		   System.out.println("Num of Vertices is too small for data - MetaData error! "
		   		+ "try running again with larger input!\n");
	       throw new RuntimeException(e);
	   } finally {
	     if (file != null) {
	       try {
	         file.close();
	         reader.close();
	       } catch (IOException e) {
	       }
	     }
	   }	   
	   return metaList;
	 } 
   
    public static void main(String args[]) 
    {
    	long totHeap = Runtime.getRuntime().totalMemory(); long freeHeap = Runtime.getRuntime().freeMemory(); 
    	int mb = 1024*1024;
    	long t0=System.currentTimeMillis();    	
    	String edgeFile = "Amazon0302.txt"; String metaFile = "amazon-meta.txt";
        int numVertices=getNumVertices(edgeFile); 
        int numThreads = 5;
        int cores = numThreads; int out=0; int sum =0;           
        int tupleBegin = 0;
        long sleepTime = 300;
       
        AdjList metaList = new AdjList(numVertices);
        metaList = createMetaDat(metaFile, numVertices); 
        AdjList adjacListParallel = new AdjList(numVertices);
        AdjList adjacList = new AdjList(numVertices);
        
        long t1=System.currentTimeMillis(); 
        adjacList = createGraph(edgeFile, numVertices);          
        int weight = getWeights(numVertices, adjacList, metaList); //out = weight;
        long t2=System.currentTimeMillis();       
        long totHeap1 = Runtime.getRuntime().totalMemory(); long freeHeap1 = Runtime.getRuntime().freeMemory();
        
        System.out.println("Here is sample graph data via Adj List: ");
        for (int i = 0; i < 3; i++)
        {     	   
     	   System.out.print("Vertex " + i + " edges are ");
             List<String> edgeList = adjacList.getTuple(Integer.toString(i));
             System.out.print(edgeList + "\n"); 
             }            
     	System.out.println("\nHere is sample meta data for Graph with SEQ weight (at end): ");        
     	for (int i = 0; i < 3; i++)
     	{   
     		System.out.print("Vertex " + i + " meta is ");
     		LinkedList<String> mList = metaList.getTuple(Integer.toString(i));
     		System.out.print(mList + "\n"); 
        }   
          
        System.out.println("\nWeights SEQ time ms= " + (t2-t1)*1);                
        System.out.println("Total # products (via Option 1 SEQ WEIGHTS) that outperform >= 0.5 neighbors is: " + weight);
        System.out.println("\nStarting Parallel Tests on " + numThreads + " Threads:"); 
        System.out.println("Building Graph via Parallel Adjacency List..."); 
      
        long t3=System.currentTimeMillis(); 
        tupleBegin = 0;
        if (cores > numVertices) cores = numVertices; //check if threads bigger than vertices and reduce if so
        int p = (numVertices/numThreads);  //get num of rows that each thread can handle for subdivision of AdjList
        int pLast = p;
        if (numVertices % numThreads != 0){ //check for odd number of rows and assign to last thread if odd num exists
          	 pLast = p + numVertices % numThreads; }
          	                        
             for (int i = 1; i <= cores; i++){  //inner-most loop by threads to get num of rows for each local thread 
                  tupleBegin = p * (i-1); //dynamically update rowStart for iteration thru AdjList
                  if (i==cores) p = pLast;
                  	
                  Thread threadInput =  new Thread(new AdjListParallelIO(tupleBegin, tupleBegin+p, 
                		  edgeFile, adjacListParallel));//call thread for graph
                  threadInput.start();              
                    
                  if (i<=cores)
      				try {
                        if (i==cores) threadInput.sleep(sleepTime);
      					threadInput.join(); //wait for thread to finish before comparing parallel output below
      					} 	catch (InterruptedException e) {
      						e.printStackTrace(); }
                  	//System.out.println("\nMy name is " + threadInput.getName() + " on Core " + (i-1));
                    }                
       
        System.out.println("\nHere is sample graph data via Parallel Adj List (Option 1): ");        
        for (int i = 0; i < 3; i++)
        {     	   
     	   System.out.print("Vertex " + i + " edges are ");
             List<String> edgeList = adjacListParallel.getTuple(Integer.toString(i));
             System.out.print(edgeList + "\n"); 
             }  
        
        System.out.println("\nGetting Parallel Weights..."); 
        if (cores > numVertices) cores = numVertices; //check if threads bigger than vertices and reduce if so
        p = (numVertices/numThreads);  //get num of rows that each thread can handle for subdivision of AdjList
        pLast = p;
        if (numVertices % numThreads != 0){ //check for odd number of rows and assign to last thread if odd num exists
        	    pLast = p + numVertices % numThreads; }
        	                        
        for (int i = 1; i <= cores; i++){  //inner-most loop by threads to get num of rows for each local thread 
                tupleBegin = p * (i-1); //dynamically update rowStart for iteration thru AdjList
                if (i==cores) p = pLast;
                	
                Thread thread =  new Thread(new AdjListParallel(out, tupleBegin, tupleBegin+p, adjacListParallel, metaList));//call thread for Weight
                thread.start();             


                if (i<=cores)
    				try {
                        if (i==cores) thread.sleep(sleepTime);
    					thread.join(); //wait for thread to finish before comparing parallel output below
    					} 	catch (InterruptedException e) {
    						e.printStackTrace(); }
                    System.out.println("\nMy name is " + thread.getName() + " on Core " + (i-1));
                    sum = sum + AdjListParallel.getOut(); //show piece-wise local output for each current thread
                  }             
        long t4=System.currentTimeMillis();  
        
        System.out.println("\nWeights PARA time ms= " + (t4-t3)*1);  
        System.out.println("Total # products (via Option 1 PARA WEIGHTS + PARA I/O) that outperform >= 0.5 neighbors is: " + sum);
        
        System.out.println("\nHere is sample meta data for Graph with both SEQ & PARA weights (at end): ");        
     	for (int i = 0; i < 3; i++)
     	{   
     		System.out.print("Vertex " + i + " meta is ");
     		LinkedList<String> mList = metaList.getTuple(Integer.toString(i));
     		System.out.print(mList + "\n"); 
        }   
        
        if (adjacList.entrySet().equals(adjacListParallel.entrySet())) System.out.println("\nSUMMARY0: SEQ & PARA [AdjList] hash tables match...");
        else System.out.println("\nSUMMARY0: ERROR: SEQ & PARA [AdjList] hash tables do not match...");
        if (sum == weight) System.out.println("SUMMARY1*: SEQ & PARA results match as total in both cases is: " + (sum+weight)/2);  
        else System.out.println("SUMMARY1*: Parallel test was unsuccessful as totals do not match! ");  
        System.out.println("SUMMARY2: ALL OPERATIONS GRAND TOTAL TIME ms= "
          + (t4-t0)*1 + "  SEQ TIME ms= " + (t2-t1) + "  PARA TIME(" + numThreads + ") ms= " 
        		+ (t4-t3) + "  METADATA TIME ms= " + ((t4-t0)-(t2-t1)-(t4-t3)) + "  MEMORY USE mb= " 
          + (((totHeap1-freeHeap1)-(totHeap-freeHeap))/mb));    
        
        System.out.println("\n*NOTE: Products with NO EDGES as well as DISCONTINUED PRODUCTS get assigned a weight of ZERO..."
        		+ "\n       If a DISCONTINUED PRODUCT is an EDGE to another product then it is assigned sales rank of INF (99999999)..."
        		+ "\n       Product IDs (0 & 9) are good examples of these cases etc...");
      
    }

}

