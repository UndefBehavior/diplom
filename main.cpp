#include <iostream>
#include <fstream>

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>

//для работы с датами
#include <boost/date_time/gregorian/gregorian.hpp>

//дабы не писать всюду std
using namespace std;

//Y`(t+1)=alpha*Y(t)+(1-alpha)*Y`(t). Возвращает пару (значение альфы,"качество" сглаживания). Принимает на вход вектор пар (дата, цена акции).
pair<double,double> get_coef_for_smoothing(const vector<pair<boost::gregorian::date,double> >& vals)
{
    double min=1;
    double coef=0;
    for(int i=1;i<100;++i)
    {
        double cur_coef=double(i)/100;
        vector<double> fc;
        long double cso=0;
        fc.emplace_back(vals[0].second);
        for(size_t j=1;j<vals.size();++j)
        {
            fc.emplace_back(cur_coef*vals[j-1].second+(1-cur_coef)*fc[j-1]);
            cso+=(vals[j].second-fc[j])*(vals[j].second-fc[j])/((fc[j])*(fc[j]));
        }
        cso/=vals.size();
        if(cso<min) {min=cso;coef=cur_coef;}
    }
    return make_pair(coef,1-min);
}

//Возвращает 0.3*h. Принимает на вход вектор пар (дата, цена акции). Использует openMP для скорости вычислений.
double get_h_for_kernel_density(const vector<pair<boost::gregorian::date,double> >& vals)
{
    double min=DBL_MAX;
    double h=0.;
    for(int i=1;i<100;++i)
    {
        double cur_h=double(i)/100;
        double cv=0.;
#pragma omp parallel for reduction(+:cv)
        for(size_t j=1;j<vals.size();++j)
        {
            long double numer=0.,denom=0.;
            for(size_t k=0;k<vals.size();++k)
            {
                if(k==j) continue;
                long double e=-(vals[j].second-vals[k].second)*(vals[j].second-vals[k].second)/(2*cur_h*cur_h);
                long double temp=exp(e);
                numer+=(sin(vals[k].second)+0.5)*temp;
                denom+=temp;
            }
            cv+=pow(vals[j].second-numer/denom,2);
        }
        cv/=vals.size();
        if(cv<min){min=cv;h=cur_h;}
    }
    return 0.3*h;
}

//Возвращает вектор пар (дата,минимум/максимум) для сглаженной функции экспоненциальным методом. Принимает на вход вектор пар фактических значений(для построения сглаженной функции) и коэффициент альфа.
//нулевой - либо минимум, либо ничего. Далее нечетный индекс - максимум, четный - минимум.
vector<pair<boost::gregorian::date,double> > get_min_max_for_exp_method(const vector<pair<boost::gregorian::date,double> >& vals, double alpha)
{
    vector<double> func;
    vector<pair<boost::gregorian::date,double> > res;
    func.emplace_back(vals[0].second);
    for(size_t j=1;j<vals.size();++j)
        func.emplace_back(alpha*vals[j-1].second+(1.-alpha)*func[j-1]);
    bool flag;//0 - последним внесен минимум; 1 - последним внесен максимум.
    if(func[1]<func[2]) {res.emplace_back(vals[1].first,func[1]);flag=0;}
    else
    {
        res.emplace_back(boost::gregorian::date(),0.);
        res.emplace_back(vals[1].first,func[1]);flag=1;
    }
    for(size_t i=2;i<func.size()-1;++i)
    {
        if(flag)
        {
            if(!((func[i-1]>func[i])&&(func[i+1]>func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
        else
        {
            if(!((func[i-1]<func[i])&&(func[i+1]<func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
    }
    if(((func[func.size()-2]<func[func.size()-1])&&(!flag))||((func[func.size()-2]>func[func.size()-1])&&flag)) res.emplace_back(vals[func.size()-1].first,func[func.size()-1]);
    return res;
}


//Возвращает вектор пар (дата,минимум/максимум) для сглаженной функции методом регрессии ядра. Принимает на вход вектор пар фактических значений(для построения сглаженной функции) и коэффициент h*.
//нулевой - либо минимум, либо ничего. Далее нечетный индекс - максимум, четный - минимум. Использует openMP для ускорения вычислений.
vector<pair<boost::gregorian::date,double> > get_min_max_for_kernel_density(const vector<pair<boost::gregorian::date,double> >& vals, double h)
{
    vector<double> func(vals.size());
    vector<pair<boost::gregorian::date,double> > res;
#pragma omp parallel for
    for(size_t j=0;j<vals.size();++j)
    {
        long double numer=0.,denom=0.;
        for(size_t k=0;k<vals.size();++k)
        {
            if(k==j) continue;
            long double e=-(vals[j].second-vals[k].second)*(vals[j].second-vals[k].second)/(2*h*h);
            long double temp=exp(e);
            numer+=(sin(vals[k].second)+0.5)*temp;
            denom+=temp;
        }
        func[j]=numer/denom;
    }
    bool flag;//0 - последним внесен минимум; 1 - последним внесен максимум.
    if(func[0]<func[1]) {res.emplace_back(vals[0].first,func[0]);flag=0;}
    else
    {
        res.emplace_back(boost::gregorian::date(),0.);
        res.emplace_back(vals[0].first,func[0]);flag=1;
    }
    for(size_t i=1;i<func.size()-1;++i)
    {
        if(flag)
        {
            if(!((func[i-1]>func[i])&&(func[i+1]>func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
        else
        {
            if(!((func[i-1]<func[i])&&(func[i+1]<func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
    }
    if(((func[func.size()-2]<func[func.size()-1])&&(!flag))||((func[func.size()-2]>func[func.size()-1])&&flag)) res.emplace_back(vals[func.size()-1].first,func[func.size()-1]);
    return res;
}


//Далее идут 10 функций, возвращающих вектор пар (дата начала, дата окончания), соответствующий всем вхождениям определенной модели в сглаженную функцию. Принимает на вход вектор пар (дата, минимум/максимум).
vector<pair<boost::gregorian::date,boost::gregorian::date> > get_hs_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i+2].second>min_maxes[i].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        double max_avg=(min_maxes[i].second+min_maxes[i+4].second)/2;
        double min_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i].second-max_avg)<=0.015*max_avg&&abs(min_maxes[i+4].second-max_avg)<=0.015*max_avg)) continue;
        if(!(abs(min_maxes[i+1].second-min_avg)<=0.015*min_avg&&abs(min_maxes[i+3].second-min_avg)<=0.015*min_avg)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_ihs_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i+2].second<min_maxes[i].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        double min_avg=(min_maxes[i].second+min_maxes[i+4].second)/2;
        double max_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i].second-min_avg)<=0.015*min_avg&&abs(min_maxes[i+4].second-min_avg)<=0.015*min_avg)) continue;
        if(!(abs(min_maxes[i+1].second-max_avg)<=0.015*max_avg&&abs(min_maxes[i+3].second-max_avg)<=0.015*max_avg)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_btop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second<min_maxes[i+2].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second>min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_bbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second>min_maxes[i+2].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second<min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_ttop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second>min_maxes[i+2].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second<min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_tbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second<min_maxes[i+2].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second>min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_rtop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        double max_avg=(min_maxes[i+2].second+min_maxes[i+4].second)/2;
        double min_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i+2].second-max_avg)<=0.0075*max_avg&&abs(min_maxes[i+4].second-max_avg)<=0.0075*max_avg)) continue;
        if(!(abs(min_maxes[i+1].second-min_avg)<=0.0075*min_avg&&abs(min_maxes[i+3].second-min_avg)<=0.0075*min_avg)) continue;
        if(!(min({min_maxes[i].second,min_maxes[i+2].second,min_maxes[i+4].second})>max({min_maxes[i+1].second,min_maxes[i+3].second}))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_rbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<5) return dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        double min_avg=(min_maxes[i+2].second+min_maxes[i+4].second)/2;
        double max_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i+2].second-min_avg)<=0.0075*min_avg&&abs(min_maxes[i+4].second-min_avg)<=0.0075*min_avg)) continue;
        if(!(abs(min_maxes[i+1].second-max_avg)<=0.0075*max_avg&&abs(min_maxes[i+3].second-max_avg)<=0.0075*max_avg)) continue;
        if(!(max({min_maxes[i].second,min_maxes[i+2].second,min_maxes[i+4].second})<min({min_maxes[i+1].second,min_maxes[i+3].second}))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_dtop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<3) return dates;
    for(size_t i=1;i<min_maxes.size()-2;i+=2)
    {
        pair<boost::gregorian::date,double> max_next={boost::gregorian::date(),DBL_MIN};
        for(size_t j=i+2;j<min_maxes.size()-1;j+=2)
            if(min_maxes[j].second>max_next.second) max_next=min_maxes[j];
        double max_avg=(min_maxes[i].second+max_next.second)/2;
        if(!(abs(min_maxes[i].second-max_avg)<=0.015*max_avg&&abs(max_next.second-max_avg)<=0.015*max_avg)) continue;
        if(!(max_next.first-min_maxes[i].first>boost::gregorian::date_duration(22))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,max_next.first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_dbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    if(min_maxes.size()<3) return dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-2;i+=2)
    {
        pair<boost::gregorian::date,double> min_next={boost::gregorian::date(),DBL_MAX};
        for(size_t j=i+2;j<min_maxes.size()-1;j+=2)
            if(min_maxes[j].second<min_next.second) min_next=min_maxes[j];
        double min_avg=(min_maxes[i].second+min_next.second)/2;
        if(!(abs(min_maxes[i].second-min_avg)<=0.015*min_avg&&abs(min_next.second-min_avg)<=0.015*min_avg)) continue;
        if(!(min_next.first-min_maxes[i].first>boost::gregorian::date_duration(22))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_next.first));
    }
    return dates;
}

//Возвращает экзогенное значение прибыли/убытка для модели. Принимает на вход пару дат модели, название модели и фактические цены.
double get_exogen_value(const pair<boost::gregorian::date,boost::gregorian::date>& model,const string& model_info,const vector<pair<boost::gregorian::date,double> >& vals)
{
    double res=0.;
    auto model_end = find_if( vals.begin(), vals.end(),
                              [&model](const pair<boost::gregorian::date,double>& element){ return element.first == model.second;} );
    auto model_after=model_end+3;
    if(model_info.find("top")!=model_info.npos)
        res=model_end->second-model_after->second;
    else res=model_after->second-model_end->second;
    return res;
}


//Возвращает эндогенное значение прибыли/убытка для модели. Принимает на вход пару дат модели, название модели и фактические цены.
double get_endogen_value(const pair<boost::gregorian::date,boost::gregorian::date>& model,const string& model_info,const vector<pair<boost::gregorian::date,double> >& vals)
{
    double res=0.;
    auto model_end = find_if( vals.begin(), vals.end(),
                              [&model](const pair<boost::gregorian::date,double>& element){ return element.first == model.second;} );
    if(model_info.find("top")!=model_info.npos)
    {
        if(model_end->second>(model_end+1)->second)
        {
            auto next_min=model_end+1;
            for(auto it=model_end+1;it!=vals.end();++it)
                if(it->second<(it-1)->second) next_min=it;
                else break;
            res=model_end->second-next_min->second;
        }
        else
        {
            auto next=model_end+1;
            for(auto it=model_end+1;it!=vals.end();++it)
                if(it->second-model_end->second<0.01*model_end->second) next=it;
                else break;
            res=model_end->second-next->second;
        }
    }
    else
    {
        if(model_end->second<(model_end+1)->second)
        {
            auto next_max=model_end+1;
            for(auto it=model_end+1;it!=vals.end();++it)
                if(it->second>(it-1)->second) next_max=it;
                else break;
            res=next_max->second-model_end->second;
        }
        else
        {
            auto next=model_end+1;
            for(auto it=model_end+1;it!=vals.end();++it)
                if(model_end->second-it->second<0.01*model_end->second) next=it;
                else break;
            res=next->second-model_end->second;
        }
    }
    return res;
}

int main()
{
    map<string,vector<pair<boost::gregorian::date,double> > > values;
    map<string,pair<double,double> > coefs;
    std::ifstream in("torg3.txt");
    //ofstream out("results.csv");
    ofstream fout("final_results.txt");
    ifstream in_csv("results.csv");
    string trash;
    cout << fixed << setprecision(6);
    //out << fixed << setprecision(6);

    //Чтение входных данных о компаниях и ценах.
    boost::gregorian::date_input_facet* facet(new boost::gregorian::date_input_facet("%d.%m.%Y"));
    in.imbue(std::locale(in.getloc(), facet));
    while(in.good())
    {
        string company="";
        in >> company;
        boost::gregorian::date price_date;
        in >> price_date;
        double price =0.;
        in >> price;
        auto place = values.find(company);
        if(place==values.end()) place=(values.emplace(company,vector<pair<boost::gregorian::date,double> >())).first;
        place->second.emplace_back(price_date,price);
    }
    values.erase("");

    //Чтение препросчитанных коэффициентов альфа и h* из results.csv. Данные препросчитаны по причине их долгого расчета (~1 час с условием работы на 4 потоках).
    getline(in_csv,trash);
    while(in_csv.good())//Company;Alpha;h*
    {
        string company(""),alpha(""),h("");
        getline(in_csv,company,';');
        getline(in_csv,alpha,';');
        getline(in_csv,h,'\n');
        if((alpha=="")||(h=="")) continue;
        coefs.emplace(company,make_pair(stod(alpha),((stod(h)==0.)?.0001:stod(h))));
    }


    cout <<"Number of companies: " << values.size() << '\n';


    for(auto& el:values)//для каждой компании
    {
        //получить вектора минимаксов.
        vector<pair<boost::gregorian::date,double> > min_maxs_e=get_min_max_for_exp_method(el.second,(coefs.find(el.first))->second.first);
        vector<pair<boost::gregorian::date,double> > min_maxs_k=get_min_max_for_kernel_density(el.second,(coefs.find(el.first))->second.second);

        //получить все реализации всех моделей для обоих методов
        map<string,vector<pair<boost::gregorian::date,boost::gregorian::date> > > models_for_exp,models_for_kernel;
        models_for_exp.emplace("hstop",get_hs_models(min_maxs_e));
        models_for_exp.emplace("ihsbot",get_ihs_models(min_maxs_e));
        models_for_exp.emplace("btop",get_btop_models(min_maxs_e));
        models_for_exp.emplace("bbot",get_bbot_models(min_maxs_e));
        models_for_exp.emplace("ttop",get_ttop_models(min_maxs_e));
        models_for_exp.emplace("tbot",get_tbot_models(min_maxs_e));
        models_for_exp.emplace("rtop",get_rtop_models(min_maxs_e));
        models_for_exp.emplace("rbot",get_rbot_models(min_maxs_e));
        models_for_exp.emplace("dtop",get_dtop_models(min_maxs_e));
        models_for_exp.emplace("dbot",get_dbot_models(min_maxs_e));

        models_for_kernel.emplace("hstop",get_hs_models(min_maxs_k));
        models_for_kernel.emplace("ihsbot",get_ihs_models(min_maxs_k));
        models_for_kernel.emplace("btop",get_btop_models(min_maxs_k));
        models_for_kernel.emplace("bbot",get_bbot_models(min_maxs_k));
        models_for_kernel.emplace("ttop",get_ttop_models(min_maxs_k));
        models_for_kernel.emplace("tbot",get_tbot_models(min_maxs_k));
        models_for_kernel.emplace("rtop",get_rtop_models(min_maxs_k));
        models_for_kernel.emplace("rbot",get_rbot_models(min_maxs_k));
        models_for_kernel.emplace("dtop",get_dtop_models(min_maxs_k));
        models_for_kernel.emplace("dbot",get_dbot_models(min_maxs_k));


        long double prof_end=0.,loss_end=0.,prof_ex=0.,loss_ex=0.;
        fout << el.first << "\nexponential method\n";
        for(auto& exp:models_for_exp)//для каждой модели экспоненциального метода
        {
            fout << exp.first << " model: total = " << exp.second.size() << '\n';//всего моделей этого типа
            size_t k=1;
            double profit=0.,loss=0.;
            if(exp.second.size())
            {
                for(auto models=exp.second.begin();models!=exp.second.end()-1;++models)//сколько моделей по годам
                {
                    if(models->second.year()==(models+1)->second.year()) ++k;
                    else
                    {
                        fout << models->second.year() << ": " << k << '\n';
                        k=1;
                    }

                }
                fout << (exp.second.end()-1)->second.year() << ": " << k << '\n';

                double profit_tot=0.,loss_tot=0.;
                fout << "endogenous method\n";
                for(auto models=exp.second.begin();models!=exp.second.end()-1;++models)//для каждого появления модели
                {
                    if(models->second.year()==(models+1)->second.year())
                    {
                        double n=get_endogen_value(*models,exp.first,el.second);
                        (n>0)? profit+=n:loss-=n;
                    }
                    else
                    {
                        profit_tot+=profit;
                        loss_tot+=loss;
                        fout << "Total in " << models->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';//вывод дохода/расхода по модели за год
                        profit=0.;
                        loss=0.;
                    }
                }
                profit_tot+=profit;
                loss_tot+=loss;
                fout << "Total in " << (exp.second.end()-1)->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';
                fout << "Total for " << exp.first << " model: profit = " << profit_tot << "; loss = " << loss_tot << '\n';//вывод дохода/расхода по модели
                prof_end+=profit_tot;
                loss_end+=loss_tot;
                profit_tot=0.;
                loss_tot=0.;
                profit=0.;
                loss=0.;
                fout << "exogenous method\n";
                for(auto models=exp.second.begin();models!=exp.second.end()-1;++models)//для каждого появления модели
                {
                    if(models->second.year()==(models+1)->second.year())
                    {
                        double n=get_exogen_value(*models,exp.first,el.second);
                        (n>0)? profit+=n:loss-=n;
                    }
                    else
                    {
                        profit_tot+=profit;
                        loss_tot+=loss;
                        fout << "Total in " << models->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';//вывод дохода/расхода по модели за год
                        profit=0.;
                        loss=0.;
                    }
                }
                profit_tot+=profit;
                loss_tot+=loss;
                fout << "Total in " << (exp.second.end()-1)->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';
                fout << "Total for " << exp.first << " model: profit = " << profit_tot << "; loss = " << loss_tot << '\n';
                prof_ex+=profit_tot;
                loss_ex+=loss_tot;
            }
        }
        fout << "profit_endogenous = " << prof_end << "\nloss_endgenous = " << loss_end << "\nprofit_exogenous = " << prof_ex << "\nloss_exogenous = " << loss_ex << '\n';//вывод общих доходов/расходов, посчитанных экзогенным/эндогенным методами, для компании по всем моделям
        prof_end=0.,loss_end=0.,prof_ex=0.,loss_ex=0.;
        fout << "\nkernel density\n";
        for(auto& kernel:models_for_kernel)//для каждой модели регрессии ядра делаем то же, что и для моделей для экспоненциального метода
        {
            fout << kernel.first << " model: total = " << kernel.second.size() << '\n';//всего моделей этого типа
            size_t k=1;
            double profit=0.,loss=0.;
            if(kernel.second.size())
            {
                for(auto models=kernel.second.begin();models!=kernel.second.end()-1;++models)//сколько моделей по годам
                {
                    if(models->second.year()==(models+1)->second.year()) ++k;
                    else
                    {
                        fout << models->second.year() << ": " << k << '\n';
                        k=1;
                    }

                }
                fout << (kernel.second.end()-1)->second.year() << ": " << k << '\n';

                double profit_tot=0.,loss_tot=0.;
                fout << "endogenous method\n";
                for(auto models=kernel.second.begin();models!=kernel.second.end()-1;++models)//для каждого появления модели
                {
                    if(models->second.year()==(models+1)->second.year())
                    {
                        double n=get_endogen_value(*models,kernel.first,el.second);
                        (n>0)? profit+=n:loss-=n;
                    }
                    else
                    {
                        profit_tot+=profit;
                        loss_tot+=loss;
                        fout << "Total in " << models->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';//вывод дохода/расхода по модели за год
                        profit=0.;
                        loss=0.;
                    }
                }
                profit_tot+=profit;
                loss_tot+=loss;
                fout << "Total in " << (kernel.second.end()-1)->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';
                fout << "Total for " << kernel.first << " model: profit = " << profit_tot << "; loss = " << loss_tot << '\n';
                prof_end+=profit_tot;
                loss_end+=loss_tot;
                profit_tot=0.;
                loss_tot=0.;
                profit=0.;
                loss=0.;
                fout << "exogenous method\n";
                for(auto models=kernel.second.begin();models!=kernel.second.end()-1;++models)//для каждого появления модели
                {
                    if(models->second.year()==(models+1)->second.year())
                    {
                        double n=get_exogen_value(*models,kernel.first,el.second);
                        (n>0)? profit+=n:loss-=n;
                    }
                    else
                    {
                        profit_tot+=profit;
                        loss_tot+=loss;
                        fout << "Total in " << models->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';//вывод дохода/расхода по модели за год
                        profit=0.;
                        loss=0.;
                    }
                }
                profit_tot+=profit;
                loss_tot+=loss;
                fout << "Total in " << (kernel.second.end()-1)->second.year() << ": profit = " << profit << "; loss = " << loss << '\n';
                fout << "Total for " << kernel.first << " model: profit = " << profit_tot << "; loss = " << loss_tot << '\n';
                prof_ex+=profit_tot;
                loss_ex+=loss_tot;
            }
        }
        fout << "profit_endogenous = " << prof_end << "\nloss_endgenous = " << loss_end << "\nprofit_exogenous = " << prof_ex << "\nloss_exogenous = " << loss_ex << '\n';
    }

    //часть,нужная для заполнения res.csv. Для ее активации надо закомментить часть, начинающуюся после values.erase("") до этого момента, раскомментить эту часть, а также строки, связанные с ofstring out. Время выполнения перезаписи ~1 часа, в зависимости от количества ядер процессора.
    /*
    out << "Company;Alpha;h*\n";
    for(auto& el:values)
    {
        cout << el.first << '\n';
        out << el.first << ';';
        pair<double,double> esc=get_coef_for_smoothing(el.second);
        cout << "esc: " << esc.first << " prob = " << esc.second << '\n';
        out << esc.first << ';';
        double h=get_h_for_kernel_density(el.second);
        cout << "h*: " << h << '\n';
        out << h.first << '\n';

    }*/
    cout << "Complete!\n";
    delete facet;
    return 0;
}

