#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct person {
    int age;
    string fullname;
};

const int myage = 42;

int main() {
    // First few lines of code using the "see-out".
    cout << "Hellow World. This is a trial.\n";
    cout << "I am learning stuff.\n\n";

    person andy;

    cout << "I am " << myage << " years old.\n";
    cout << "How old is Andy?\n";
    cin >> andy.age;
    cin.ignore();
    cout << "\nAndy is " << andy.age << " years old.\n\n";
    
    cout << "What is Andy's fullname?\n";
    getline(cin, andy.fullname);
    cout << "So, Andy's fullname is " << andy.fullname << ". Who knew?\n\n";

    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg) { // a loop for fun
        cout << word << " ";
    }
    cout << endl;
}