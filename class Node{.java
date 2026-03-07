class Node{
    int data;
    Node next;
    Node (int val){
        data =val;
        next = null;
    }
}

public class practice {
    static Node head;
    static Node temp;
    
    static void insertAtHead(int data){
        Node newnode = new Node(data);
        if(head == null ){
            head = newnode;
            return;
        }
        newnode.next=head;
        head = newnode;
    }

    void instertAtTail(int data)
    {
        Node newnode = new Node(data);
        if(head == null){
            head = newnode;
            return;
        }
        Node temp = head;
        while(temp.next!=null){
            temp = temp.next;
        }
        temp.next = newnode;
    }

    void deleteAtHead(){
        if(head == null){
            System.out.println("List is empty");
            return;
        }
        head = head.next;
    }
    
    static void display(){
        while(temp!=null){
            System.out.print(temp.data+" ");
            temp = temp.next;
        }
    }
    
    public static void main(String[] args) {
        Node head = new Node(10);
        Node h2 = new Node(20);
        Node h3 = new Node(30);
        head.next=h2;
        h2.next=h3;
        temp = head;
        display();
    }
}