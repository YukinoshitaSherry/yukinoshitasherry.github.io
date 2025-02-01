---
title: 面向对象程序设计期末复习题习题集
date: 2023-01-08 15:00:00
categories: 
- 上浙大
tags:
- 专业必修
- 试卷习题
desc: 2023~2024秋冬期末复习，CZP老师班OOP复习题5个习题集摘录
---

## 习题集1
### 判断题
#### 1-1
对单目运算符重载为友元函数时，可以说明一个形参。而重载为成员函数时，不能显式说明形参。(T)
【解答】：回忆知识点，双目重载为友元2个形参，重载为成员1个形参(还有一个参数默认是this指针)；单目重载为友元1个形参，重载为成员0个形参(一个参数默认是this指针)
#### 1-2
虚函数是用virtual 关键字说明的成员函数。(T)

#### 1-3
使用提取符(<<)可以输出各种基本数据类型的变量的值，也可以输出指针值。(T)

#### 1-4
因为静态成员函数不能是虚函数，所以它们不能实现多态。(F)

#### <mark style="background: #FFF3A3A6;">1-5</mark>
重载operator+时，返回值的类型应当与形参类型一致。  
比如以下程序中，operator+的返回值类型有错：
```c++
class A {
int x;
public:
 A(int t=0):x(t){ }
    int operator+(const A& a1){ return x+a1.x;  }
};
```
（F）
【解答】`operator+`的返回类型并不需要与参数类型一致。实际上`operator+`通常返回结果的类型可能与参数类型完全不同。


### 单选题
#### 2-1
对象之间的相互作用和通信是通过消息。（ ）不是消息的组成部分。
A.接受消息的对象
B.要执行的函数的名字
<mark style="background: #FFF3A3A6;">C</mark>.要执行的函数的内部结构
D.函数需要的参数


#### 2-2
（ ）不是面向对象程序设计的主要特征。
A.封装
B.继承
C.多态
<mark style="background: #FFF3A3A6;">D</mark>.结构

#### 2-3
对定义重载函数的下列要求中，（ ）是错误的。
A.要求参数的个数不同
B.要求参数中至少有一个类型不同
<mark style="background: #FFF3A3A6;">C</mark>.要求函数的返回值不同
D.要求参数个数相同时，参数类型不同


#### 2-4
关于动态绑定的下列描述中，（ ）是错误的。
A.动态绑定是以虚函数为基础的
B.动态绑定在运行时确定所调用的函数代码
C.动态绑定调用函数操作是通过指向对象的指针或对象引用来实现的
<mark style="background: #FFF3A3A6;">D</mark>.动态绑定是在编译时确定操作函数的
【解答】**运行**时确定的

#### <mark style="background: #FFF3A3A6;">2-5</mark>
关于虚函数的描述中，（ ）是正确的。
A.虚函数是一个static 类型的成员函数
B.虚函数是一个非成员函数
C.基类中说明了虚函数后，派生类中与其对应的函数可不必说明为虚函数
D.派生类的虚函数与基类的虚函数具有不同的参数个数和类型
【解答】虚函数是成员函数；基类的虚函数无论被公有继承多少次，在多级派生类中**仍然**为虚函数；派生类的虚函数与基类的虚函数具有相同的参数个数和类型
#### 2-6
下列叙述中，不正确的是（ ）。
A.构造函数必须和类同名
B.构造函数和析构函数都没有返回值
<mark style="background: #FFF3A3A6;">C</mark>.析构函数中不能加代码
D.析构函数不能带参数


#### 2-7
在下面类声明中，关于生成对象不正确的是（ ）。  
```c++
class point  
{ public:  
int x;  
int y;  
point(int a,int b) {x=a;y=b;}  
};
```

A.`point p(10,2);`
B.`point *p=new point(1,2);`
<mark style="background: #FFF3A3A6;">C</mark>.`point *p=new point[2];`
D.`point *p[2]={new point(1,2), new point(3,4)};`

#### 2-8
下列运算符中，（ ）运算符不能重载。
A.`＆`
B.`[ ]`
<mark style="background: #FFF3A3A6;">C</mark>.`::`
D.`<<`

#### 2-9
关于纯虚函数和抽象类的描述中，（ ）是错误的。
A.纯虚函数是一种特殊的虚函数，它没有具体的实现
B.抽象类是指具有纯虚函数的类
<mark style="background: #FFF3A3A6;">C</mark>.一个基类中说明有纯虚函数，该基类的派生类一定不再是抽象类
D.抽象类只能作为基类来使用，其纯虚函数的实现由派生类给出
【解答】纯虚函数是在声明虚函数时被“初始化”为0的虚函数。不定义对象而只作为一种基本类型作为继承的类,称为抽象类。凡是包含纯虚函数的类都是抽象类。抽象类的作用是作为一个类族的共同基类。

#### 2-10
在下列关键字中,用以说明类中公有成员的是（ ）。
<mark style="background: #FFF3A3A6;">A.
</mark>public
B.private
C.protected
D.friend

### 程序填空题
#### 5-2
CAT's Copy
阅读下面的程序，完成其中复制构造函数的代码。
```C++
#include <iostream>
using namespace std;
class CAT
{     public:
           CAT();
           CAT(const CAT&);
          ~CAT();
          int GetAge() const { return *itsAge; }
          void SetAge(int age){ *itsAge=age; }
      protected:
          int* itsAge;
};
CAT::CAT()
{    itsAge=new int;
     *itsAge =5;
}
CAT::CAT(const CAT& c)
{
/*______5 分;*/
/*______5 分;*/
}
CAT::~CAT()
{  delete itsAge; }
```
- 【解答】
	- `itsAge = new int; // 分配新的内存空间`
	- `*itsAge = *(c.itsAge); // 复制传入对象的itsAge的值`



## 习题集2
### 填空题
#### 1
write the output of the code below.
```C++
#include<iostream>
using namespace std;

class INCREMENT 
{
public:
   INCREMENT( int v = 0, int i = 1 ); 
   void addIncrement() 
   { 
      v += increment; 
   } 
   void print() const; 
   int get() const
   {
       return v;
   }
private:
   int v;
   const int increment; 
}; 

INCREMENT::INCREMENT( int v, int i ) : v( v ), increment( i )    
{ 
} 

void INCREMENT::print() const
{
   cout << v << endl;
} 
int main()
{
   INCREMENT value( 1, 2);
   value.print();
   for ( int j = 1; j <= 2; j++ ) 
   {
      value.addIncrement();
      value.print();
   } 
   return 0;
}
```

- 【解答】
	- 1.`1`
	- 2.`3`
	- 3.`5`

#### 2
write the output of the code below.
```C++
#include<iostream>
using namespace std;
class TEST
{ 
    int num; 
public:
    TEST( int num=0); 
    void increment( ) ;
    ~TEST( );
}; 
TEST::TEST(int num) : num(num)
{
    cout << num  << endl; 
} 
void TEST::increment() 
{
    num++;
}
TEST::~TEST( )
{
    cout << num  << endl;
} 
int main( ) 
{
    TEST array[2]; 
    array[0].increment();
    array[1].increment();
    return 0;
}
```
- 【解答】
	- 1.`0`
	- 2.`0`
	- 3.`1`
	- 4.`1`

## 习题集3
### 填空题
#### 1
write the output of the code below.
```c++
#include <iostream>
using namespace std;
int& f(int &i )
{
    i += 10;
    return i ;
}
int main()
{
    int k = 0;
    int& m = f(k);
    cout << k << "#";
    f(m)++;
    cout << k << endl;
    return 0;
}
```
- 【解答】`10#21`
	- 在`main`函数中，定义了一个整数`k`并初始化为0
	- 然后，它调用函数`f`，将`k`作为引用参数传入。在函数`f`中，参数`i`是对`k`的引用，所以`i += 10;`实际上是将`k`的值增加10。所以，`k`的值现在是10。函数`f`返回对`k`的引用，这个引用被赋给了`m`
	- `cout << k << "#";`将输出`10#`
	- 接下来，`f(m)++`再次调用了函数`f`，这次传入的是`m`，<mark style="background: #FFF3A3A6;">`m`实际上是对`k`的引用</mark>，所以`i += 10;`再次将`k`的值增加10，`k`的值现在是20。函数`f`返回对`k`的引用，然后`++`操作将`k`的值增加1，所以`k`的值现在是21
		- 高亮解释<img src="https://raw.githubusercontent.com/YukinoshitaSherry/qycf_picbed/main/img/20250201170316198.png" alt="23oop复习题-1" style="width:90%;" />

	- 最后，`cout << k << endl;`将输出`21`
#### 2
write the output of the code below.
```c++
#include <bits/stdc++.h>
using namespace std;
class counter {
  private:
    int value;
  public:
    counter() : value(0) {}
    counter &operator++();
    int operator++(int);
    void reset() { value = 0; }
    operator int() const { return value; }
};

counter &counter::operator++() {
    if (3 == value) value = 0;
    else
        value += 1;
    return *this;
}

int counter::operator++(int) {
    int t = value;
    if (3 == value) value = 0;
    else
        value += 1;
    return t;
}

int main() {
    counter a;
    while (++a) cout << "***\n";
    cout << a << endl;
    while (a++) cout << "***\n";
    cout << a << endl;
    return 0;
}
```
- 【答案】
	- 1.`***`
	- 2.`***`
	- 3.`***`
	- 4.`0`
	- 5.`1`
- 【解答】
	- 最初a=0，++a进入`counter &counter::operator++()`前置递增
	- 第一次 `++a`，a=1,输出***
	- 第二次 `++a`，a=2,输出***
	- 第三次 `++a`<mark style="background: #FFF3A3A6;">后</mark>，a=3,输出***
	- 第四次 `++a`，a=0，不进去
	- a=0,输出0
	- a++进入`int counter::operator++(int)` 后置递增
	- 第一次 `a++`，while(a)中a=0不进去,然后a自增=1
	- a=1,输出1

#### 3
```c++
#include <bits/stdc++.h>
using namespace std;
using namespace std;
class Sample {
    friend long fun(Sample s);
  public:
    Sample(long a) { x = a; }
  private:
    long x;
};
long fun(Sample s) {
    if (s.x < 2) return 1;
    return s.x * fun(Sample(s.x - 1));
}
int main() {
    int sum = 0;
    for (int i = 0; i < 6; i++) {
        sum += fun(Sample(i));
    }
    cout << sum;
    return 0;
}
```
- 输出 `154`
	- 自己算一下


## 习题集4
### 填空题
#### 1
write the output of the code below.
1.the output at //1 is 
2.the output at //2 is 
3.the output at //3 is 
4.the output at //4 is 
5.the output at //5 is 
```c++
#include <iostream>
#include <string>
using namespace std ;
class Testing
{
private:
    string words; 
    int number ;
public:
    Testing(const string & s = "Testing")
    {
        words = s ;
        number = words.length();
        if (words.compare("Testing")==0)
            cout << 1;
        else if (words.compare("Heap1")==0)
            cout << 2;
        else
            cout << 3;
    }
    ~Testing()
    {
        cout << 0;
    }
    void show() const
    {
        cout << number;
    }
};
int main()
{
    Testing *pc1 , *pc2;
    pc1 = new Testing ;          //1
    pc2 = new Testing("Heap1");  //2
    pc1->show();   //3
    delete pc1 ;   //4
    delete pc2 ;   //5
    return 0;
}
```
- 【答案】
	- 1.`1`
	- 2.`2`
	- 3.`7`
	- 4.`0`
	- 5.`0`
- 【解析】
	- 注意第三题是7，length不包括`\0`

#### 2
write the output of the code below.
```c++
#include<iostream>
#include<string>
using namespace std;
class Pet {
public:
    virtual string speak() const { return "pet!"; }
};
class Dog : public Pet {
public:
    string speak() const { return "dog!"; }
};
int main() {
    Dog ralph;
    Pet* p1 = &ralph;
    Pet& p2 = ralph;
    Pet p3;
    cout << p1->speak() <<endl;
    cout << p2.speak() << endl;
    cout << p3.speak() << endl;
    return 0;
}
```
- 【答案】
	- 1.`dog!`
	- 2.`dog!`
	- 3.`pet!`
		- 注意是纯虚函数不能实例化，虚函数是可以的！


#### 3
重复 过
#### 4
write the output below
```c++
#include <iostream>
using namespace std;
enum NOTE { middleC, Csharp, Cflat };
class Instrument {
  public:
    virtual void play(NOTE) const = 0;
    virtual char *what() const = 0;
    virtual void adjust(int) = 0;
};
class Wind : public Instrument {
  public:
    void play(NOTE) const { cout << 1 << endl; }
    char *what() const { return "Wind"; }
    void adjust(int) {}
};
class Percussion : public Instrument {
  public:
    void play(NOTE) const { cout << 2 << endl; }
    char *what() const { return "Percussion"; }
    void adjust(int) {}
};
class Stringed : public Instrument {
  public:
    void play(NOTE) const { cout << 3 << endl; }
    char *what() const { return "Stringed"; }
    void adjust(int) {}
};
class Brass : public Wind {
  public:
    void play(NOTE) const { cout << 11 << endl; }
    char *what() const { return "Brass"; }
};
class Woodwind : public Wind {
  public:
    void play(NOTE) const { cout << 12 << endl; }
    char *what() const { return "Woodwind"; }
};
void tune(Instrument &i) { i.play(middleC); }
void f(Instrument &i) { i.adjust(1); }
int main() {
    Wind flute;
    Percussion drum;
    Stringed violin;
    Brass flugelhorn;
    Woodwind recorder;
    tune(flute);
    tune(drum);
    tune(violin);
    tune(flugelhorn);
    tune(recorder);
    f(flugelhorn);
    return 0;
}
```
- 【答案】
	- 1.`1`
	- 2.`2`
	- 3.`3`
	- 4.`11`
	- 5.`12`

#### 5
重复，过

#### <mark style="background: #FFF3A3A6;">6</mark>
```c++
#include <iostream>
using namespace std;
class Base {
  protected:
    int x;

  public:
    Base(int b = 0) : x(b) {}
    virtual void display() const { cout << x << endl; }
};
class Derived : public Base {
    int y;

  public:
    Derived(int d = 0) : y(d) {}
    void display() { cout << x << "," << y << endl; }
};
int main() {
    Base b(1);
    Derived d(2);
    Base *p = &d;
    b.display();
    d.display();
    p->display();
    return 0;
}
```
- 【解答】
	- 1.`1`
	- 2.`0,2`
	- 3.`0`
- 【解析】
	- `b.display()`调用的是`Base`类的`display()`函数，因为`b`是`Base`类型的对象。`Base`类的`display()`函数会输出`x`的值，所以这行代码会输出`1`。
	- `d.display()`调用的是`Derived`类的`display()`函数，因为`d`是`Derived`类型的对象。`Derived`类的`display()`函数会输出`x`和`y`的值。这里需要注意的是，尽管`Derived`类没有显式地初始化`x`，但是`x`会被`Base`类的构造函数初始化为`0`，而`y`会被`Derived`类的构造函数初始化为`2`。所以这行代码会输出`0,2`。
	- `Derived`类的`display()`函数并没有被声明为`override`，并且它的函数签名与`Base`类的`display()`函数不同，因为它不是`const`函数。所以，实际上`Derived`类并没有重写`Base`类的`display()`函数，而是定义了一个新的`display()`函数。当`p->display()`被调用时，由于`p`是一个`Base`类型的指针，所以会调用`Base`类的`display()`函数，而不是`Derived`类的`display()`函数。`Base`类的`display()`函数会输出`x`的值，由于`p`指向的是`Derived`对象`d`，而`d`的`x`值在`Derived`的构造函数中并未被显式初始化，所以它的值是由`Base`的构造函数初始化的，为`0`。
		- 对第三行的错误解释(错的！)：`p->display()`调用的是`Derived`类的`display()`函数，因为`p`是一个指向`Derived`对象的`Base`类型的指针。C++中，当通过基类的指针或引用调用一个虚函数时，会根据指针或引用实际所指向的对象类型来决定调用哪个函数。
- <mark style="background: #FFF3A3A6;">【改编】</mark>
	- 如果`virtual void display() { cout << x << endl; }`改成`virtual void display() const { cout << x << endl; }`
		- 输出就是`1`,`0,2`,`0,2`
	- 如果`void display() { cout << x << "," << y << endl; }`改成`void display() overide { cout << x << "," << y << endl; }`
		- 输出就是`1`,`0,2`,`0,2`
	- 如果同时改上面2个会报错



## 习题集5
### 填空题
#### 5
write the output of the code below.
```c++
#include<iostream>
using namespace std;
class A{   
public:
    A& operator=(const A& r)
    {
        cout << 1 << endl;
        return *this;
    }
};
class B{   
public:
    B& operator=(const B& r)
    {
        cout << 2 << endl;
        return *this;
    }
};
class C{
private:
    B b;
    A a;
    int c;
};

int main()
{
    C m,n;
    m = n;
    return 0;
}
```
- 【答案】
	- 1.`2`
	- 2.`1`
- 【解答】
	- 这段代码中，`C`类包含了`A`类和`B`类的对象，以及一个`int`类型的变量。在`main`函数中，使用了`C`类对象之间的赋值操作`m = n;`。由于`C`类没有定义自己的赋值运算符`operator=`，所以编译器会为`C`类生成一个默认的赋值运算符。
	- 默认的赋值运算符会按照成员在类中声明的顺序，依次调用每个成员的赋值运算符。在`C`类中，成员的声明顺序是`B b, A a, int c`，所以在执行`m = n;`时，首先调用`B`类的赋值运算符，然后调用`A`类的赋值运算符，最后赋值`int`类型的成员。
	- `B`类的赋值运算符输出数字`2`，`A`类的赋值运算符输出数字`1`，所以程序的输出结果是`2 1`。而`int`类型的赋值操作不会产生输出，所以没有其他输出。

#### 6
write the output of the code below.
```c++
#include <iostream>  
using namespace std;
class MYCLASS{
public:
    MYCLASS(int x):val(x){}
    void print() const
    {
        cout << val << endl;
    }
    void print()
    {
        cout << val << endl;
    }
private:
    int val;
}; 
int main()
{
   MYCLASS ob1(1);
   const MYCLASS ob2(2);
   ob2.print();
   ob1.print();
   return 0;
}
```
- 【答案】
	- 1.`2`
	- 2.`1`
