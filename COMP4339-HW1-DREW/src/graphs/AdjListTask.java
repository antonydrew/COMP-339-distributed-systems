package graphs;

import java.io.*;
import java.util.*;
import graphs.AdjListTaskParallel;
import message.*;
import java.lang.reflect.*;


public class AdjListTask extends LinkedHashMap<String, LinkedList<String>>
{
   
   private static final long serialVersionUID = 1L;
   private  Map<String,LinkedList<String>> AdList;	
   
   public AdjListTask(int numVertices)
   {
	 	
	    AdList = (new LinkedHashMap<String, LinkedList<String>>());		
        for (int i = 0; i < numVertices;  i++)
        {
           AdList.put(Integer.toString(i), new LinkedList<String>());
        }
   }
   
   public void addEdge(String src, String desty, String task)//, String task)
   {

       List<String> srcList = AdList.get(src);
       srcList.add(desty);
       srcList.add(task);
      
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
   
   public int getTaskUnit(String src, AdjListTask adjacList)
   {     
       
       List<String> eList = adjacList.getTuple(src);

       int unit = 0;
       String task = eList.get(1);
       
       String[] values = new String [100];	   
	    
	   values= task.split("-");
	      
	   task = values[1]; 
	   unit = Integer.parseInt(task);
       
       return unit;
    }
   
   public String getTaskName(String src, AdjListTask adjacList)
   {     
       
       List<String> eList = adjacList.getTuple(src);

       String task = eList.get(1);
       
       String[] values = new String [100];	   
	    
	   values= task.split("-");
	      
	   task = values[0]; 
	   
       
       return task;
    }
   
   public static int getSumVertices(int numVertices, AdjListTask adjacList) 
	{     
	   int totalWeight = 0; 
	   //int numVertices = adjacList.getTaskUnit(src, adjacList);
	  	
	   for (int k = 0; k < numVertices-1; k++)
       {		  	   
//		List<String> eList = adjacList.getTuple(Integer.toString(k));
//	    int elen = eList.size();
//	    System.out.println(elen);      

        totalWeight = totalWeight + k;       
       }	   
       return totalWeight;   	   
	}     
   
   
   
   public static int getSumTasks(int message1, int message2, int message3, AdjListTask adjacList) 
  	{     
  	   int totalWeight = 0; 
  	   String taskName = adjacList.getTaskName(Integer.toString(message3), adjacList); 
  	   
  	   int num1 = adjacList.getTaskUnit(Integer.toString(message1), adjacList);  
  	   int sum1 = adjacList.getSumVertices(num1+1, adjacList);
  	   
  	   int num2 = adjacList.getTaskUnit(Integer.toString(message2), adjacList);
	   int sum2 = adjacList.getSumVertices(num2+1, adjacList);
  	   
	   if (taskName!="sleep") totalWeight = sum1 + sum2;
	   else totalWeight = 0;
  	  	  
         
  	  	return totalWeight;   	   
  	}  
   
   public static int getSumTasksPara(int message1, int message2, int message3, AdjListTask adjacList) 
 	{     
 	   int totalWeight = 0; 
 	   String taskName = adjacList.getTaskName(Integer.toString(message3), adjacList); 
 	   
 	   
	   if (taskName!="sleep") totalWeight = message1 + message2;
	   else totalWeight = 0;
 	  	  
        
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
	     while ((line = reader.readLine()) != null) {
	       values= line.split("\t");
	       //counter++;
	       if (counter >=0) {
	    	   numVertices = Integer.parseInt(values[0]);
	    	   
	       }
	       
	     }
	  
	   } catch (Exception e) {
		   
	   } finally {
	     if (file != null) {
	       try {
	         file.close();
	         reader.close();
	       } catch (IOException e) {
	       }
	     }
	   }
	   
	   return numVertices+1;
	 } 
     
   public static AdjListTask createGraph(String fileName, int numVertices) {
	   
	   int counter = 1; String src, desty, task; 
	   AdjListTask adjacList = new AdjListTask(numVertices);
	   FileReader file = null;
	   BufferedReader reader = null;
	   String[] values = new String [numVertices*numVertices];	   
	   
	   try 
	   {
	     file = new FileReader(fileName);
	     reader = new BufferedReader(file);
	     String line = "";
	     while ((line = reader.readLine()) != null) {
	       values= line.split("\t");
	       if (counter >= 0) {
	    	   src = values[0]; desty = values[1]; task = values[2];
	    	   if ( src == desty)
	    	   {
	    		   System.out.println("Cycle detected in graph - not allowed");  //System.out.println(desty);
	    		   break;
	    	   }
	    	   if ( src != desty) adjacList.addEdge(src,desty,task); //System.out.println(desty);
	       }
	       counter++;
	     }
	   } catch (Exception e) {

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
 
   
    public static void main(String args[]) 
    {
    
    	long totHeap = Runtime.getRuntime().totalMemory(); long freeHeap = Runtime.getRuntime().freeMemory(); 
    	int mb = 1024*1024;
    	long t0=System.currentTimeMillis();    	
    	String edgeFile = "Tasks.txt"; 
        int numVertices=getNumVertices(edgeFile); 
        int numThreads = numVertices;
        int cores = numThreads; int out=0; int sum =0;  int psum = 0;         
        int tupleBegin = 0; 
       
        
        AdjListTask adjacList = new AdjListTask(numVertices);
        
        long t1=System.currentTimeMillis(); 
        adjacList = createGraph(edgeFile, numVertices);          
        int weight = getSumVertices(numVertices+1, adjacList); //out = weight;
      
        
        long t2=System.currentTimeMillis();       
        long totHeap1 = Runtime.getRuntime().totalMemory(); long freeHeap1 = Runtime.getRuntime().freeMemory();
        
        System.out.println("Here is sample graph data via Adj List: ");        
        for (int i = 0; i < 10; i++)
        {     	   
     	   System.out.print("Vertex " + (i) + " edges and tasks [edge num + task des] are: ");
             List<String> edgeList = adjacList.getTuple(Integer.toString(i));
             System.out.print(edgeList + "\n"); 
             }   
              
                      
        System.out.println("\nSum of Vertices is: " + weight);
        System.out.println("Sum SEQ time ms= " + (t2-t1)*1);  
        System.out.println("\nStarting SEQ & Parallel Execution Tests on " + numThreads + " Threads:"); 
      
        long t3=System.currentTimeMillis(); 
        tupleBegin = 0;
        if (cores > numVertices) cores = numVertices; //check if threads bigger than vertices and reduce if so
        int p = (numVertices/numThreads);  //get num of rows that each thread can handle for subdivision of AdjList
        int pLast = p;

        
        if (cores > numVertices) cores = numVertices; //check if threads bigger than vertices and reduce if so
        p = (numVertices/numThreads);  //get num of rows that each thread can handle for subdivision of AdjList
        pLast = p; String taskName = "test"; int unitp = 0;
        if (numVertices % numThreads != 0){ //check for odd number of rows and assign to last thread if odd num exists
        	    pLast = p + numVertices % numThreads; }
        	                        
        for (int i = 0; i < cores; i++){  //inner-most loop by threads to get num of rows for each local thread 
                tupleBegin = p * (i-1); //dynamically update rowStart for iteration thru AdjList
                if (i==cores) p = pLast;
                	
                
                taskName = adjacList.getTaskName(Integer.toString(i), adjacList);            
                unitp = adjacList.getTaskUnit(Integer.toString(i), adjacList);
                
                if (taskName == "sleep")
					try {
						Thread.sleep(unitp); 
					} catch (InterruptedException e1) {
						continue;
					}
                
                System.out.println("My name is SEQ " + Thread.currentThread());
                
                if (i==2 || i==3) {  //Parallel Summing: (B>>>C or D>>>E)
                	Thread thread =  new Thread(new AdjListTaskParallel(out, tupleBegin, adjacList.getTaskUnit(Integer.toString(i), adjacList), Integer.toString(i),adjacList));//call thread for Weight
                    thread.start();  
                    System.out.println("My name is PARA " + thread.getName() + " on Core " + (i-0));
                    if (i==2 || i==3)
                    {
                    
                    	try {
                    		thread.join();
                    	} catch (InterruptedException e) {
                    		continue;
                    	}
                    	
                    	if (i==3)
                        {
                        int sumTasksPara = getSumTasksPara(psum, AdjListTaskParallel.getOut(), i, adjacList) ;
                        System.out.println("\nSum of Tasks Para is: " + sumTasksPara);
                        }
                    
                    psum = psum + AdjListTaskParallel.getOut(); //get piece-wise local output for each current thread
                    }
                    
                    
                    
                }
                
                	
                               
                    if (i!=2 || i!=3) sum = sum + i;
                    System.out.println("Current thread task is: " + taskName +" w/task unit of " + unitp);
           }  
        
        int sumTasksSeq = getSumTasks(2, 3, 3, adjacList) ;
        System.out.println("\nSum of Tasks Seq is: " + sumTasksSeq);
        
        long t4=System.currentTimeMillis();  

        System.out.println("\nSum of Vertices is: " + sum);
        System.out.println("Sum PARA time ms= " + (t4-t3)*1);  
        

        System.out.println("\nSUMMARY1: SEQ & PARA results match as total in both cases is: " + (sum+weight)/2);  
        System.out.println("SUMMARY2: ALL OPERATIONS GRAND TOTAL TIME ms= "
          + (t4-t0)*1 + "  SEQ TIME(" + numThreads + ") ms= " + (t2-t1) + "  PARA TIME(" + numThreads + ") ms= " 
        		+ (t4-t3) + "  METADATA TIME ms= " + ((t4-t0)-(t2-t1)-(t4-t3)) + "  MEMORY USE mb= " 
          + (((totHeap1-freeHeap1)-(totHeap-freeHeap))/mb));    
        
        
                
    }

}

