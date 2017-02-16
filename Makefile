main: Lab2.java
	javac Lab2.java

wayne: Lab2W.java
	javac Lab2W.java

protein: Lab2W.class
	java Lab2W data/protein-secondary-structure.data

clean:
	rm -f *.class
